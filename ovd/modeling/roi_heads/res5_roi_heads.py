from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
from .custom_fast_rcnn import CustomFastRCNNOutputLayers
from detectron2.layers import ShapeSpec, cat, nonzero_tuple
import torch
@ROI_HEADS_REGISTRY.register()
class CustomRes5ROIHeads(Res5ROIHeads):
    @configurable
    def __init__(self, **kwargs):
        cfg = kwargs.pop('cfg')
        super().__init__(**kwargs)
        stage_channel_factor = 2 ** 3
        out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor

        self.with_image_labels = cfg.WITH_IMAGE_LABELS
        self.ws_num_props = cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS
        self.with_region_to_image = cfg.DDPM.WITH_REGION_TO_IMAGE
        self.box_predictor = CustomFastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )
        self.with_ddpm_reconstruction = cfg.MODEL.ROI_BOX_HEAD.USE_DDPM and cfg.DDPM.WITH_RECONSTRUCTION
        self.with_cond_noise = cfg.MODEL.ROI_BOX_HEAD.USE_DDPM and cfg.DDPM.WITH_COND_NOISE

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['cfg'] = cfg
        return ret

    def forward(self, images, features, proposals, targets=None, ann_type='box'):
        del images
        features, distill_clip_features = features
        if self.training:
            if ann_type == 'box':
                proposals = self.label_and_sample_proposals(proposals, targets)
            else:
                proposals = self.get_top_proposals(proposals)

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform([features[f] for f in self.in_features], proposal_boxes)
        predictions = self.box_predictor(box_features.mean(dim=[2, 3])) # CustomFastRCNNOutputLayers

        if self.training and distill_clip_features is not None:

            clip_img_proposals, clip_img_embed = distill_clip_features
            obj_embed = self._shared_roi_transform([features[f] for f in self.in_features], clip_img_proposals) 
            obj_embed = obj_embed.mean(dim=[2, 3])

            distill_features = (obj_embed, clip_img_embed)
        else:
            distill_features = None

        if self.training:
            loss_dict ={}
            del features
            if ann_type != 'box': # imgage-level supervisions
                image_labels = [x._pos_category_ids for x in targets]
                losses = self.box_predictor.image_label_losses(
                    predictions, proposals, distill_features, image_labels)
            
                loss_dict.update(losses)
                loss_dict.update({
                    'obj_to_txt_gen_loss': predictions[0].new_zeros([1])[0],
                    
                })
                if self.with_cond_noise:
                    loss_dict['kl_loss'] = predictions[0].new_zeros([1])[0]

            else: # instance-level supervisions
                # detector loss + obj_to_img_loss
                losses = self.box_predictor.losses(
                    (predictions[0], predictions[1]), proposals, distill_features)
                
                loss_dict.update(losses)
                
                # obj_to_txt_gen_loss
                if self.with_ddpm_reconstruction:
                    gt_classes = (
                                cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
                            )                   
                    loss_obj_to_txt = self.box_predictor.cls_score.obj_to_txt_diff(box_features.mean(dim=[2, 3]), gt_classes)
   
                    loss_dict.update(loss_obj_to_txt)
    
                if self.with_image_labels:
                    assert 'pms_loss' not in losses
                    loss_dict['pms_loss'] = predictions[0].new_zeros([1])[0]
            return proposals, loss_dict
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def get_top_proposals(self, proposals):
        for i in range(len(proposals)):
            proposals[i].proposal_boxes.clip(proposals[i].image_size)
        proposals = [p[:self.ws_num_props] for p in proposals]
        return proposals