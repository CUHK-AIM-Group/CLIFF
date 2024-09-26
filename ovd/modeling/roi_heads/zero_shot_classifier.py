import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from .diffusion_lib import DiffusionMLP, GaussianDiffusion, VaeHiddenLayer
from .gradient_scalar_layer import GradientScalarLayer

class ZeroShotClassifier(nn.Module):
    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            num_classes: int,
            zs_weight_path: str,
            zs_weight_dim: int = 512,
            use_bias: float = 0.0,
            norm_weight: bool = True,
            norm_temperature: float = 50.0,
            use_ddpm: bool = False,
            num_steps_region_to_text: int = 10,
            num_steps_region_to_image: int = 3,
            norm_ddpm_sampling: str = 'clamp',
            num_bottleneck_layers: int = 2,
            hidden_dim: int = 256,
            ctx_dim: int = 0,
            with_cond_noise: bool = False,
            with_region_to_image: bool = False,
            rec_loss_weight: float = 1.0,
            rec_loss_l1: bool = False,
            with_middle_cond_noise: bool = False,
            refine_cond_type: str = 'att',
            dual_forward: bool = False,
            kl_loss_weight: float = 1.0,
            with_over_sampling: bool = False,

    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature
        self.use_ddpm = use_ddpm
        self.norm_ddpm_sampling = norm_ddpm_sampling
        self.use_bias = use_bias < 0
        self.with_cond_noise = with_cond_noise
        self.with_region_to_image = with_region_to_image
        self.num_steps_region_to_text = num_steps_region_to_text
        self.rec_loss_weight = rec_loss_weight
        self.rec_loss_l1 = rec_loss_l1
        self.with_middle_cond_noise = with_middle_cond_noise
        self.refine_cond_type = refine_cond_type
        self.dual_forward = dual_forward
        self.kl_loss_weight = kl_loss_weight
        self.with_over_sampling = with_over_sampling

        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.linear = nn.Linear(input_size, zs_weight_dim)
        if zs_weight_path.split('.')[-1] == 'pt':
            print('use prompt distribution')
            self.with_prompt_dist = True
            zs_weight = torch.load(zs_weight_path).permute(0,2, 1).contiguous().cuda().float()  # 63 x 512 x C
            zs_weight = torch.cat(
                [zs_weight, zs_weight.new_zeros((zs_weight.size(0),zs_weight.size(1), 1))], # 63 x 512 x C + 1
                dim=-1)  # D x (C + 1)
            zs_weight = F.normalize(zs_weight, p=2, dim=1)
            self.num_prompts, self.dim, self.num_classes = zs_weight.size()

        else:
            print('use single prompt')
            zs_weight = torch.tensor(
                np.load(zs_weight_path),
                dtype=torch.float32).permute(1, 0).contiguous()  # D x C
            self.with_prompt_dist = False

            zs_weight = torch.cat(
                [zs_weight, zs_weight.new_zeros((zs_weight_dim, 1))],
                dim=1)  # D x (C + 1)
            zs_weight = F.normalize(zs_weight, p=2, dim=0)

            self.dim, self.num_classes = zs_weight.size()

        if with_cond_noise:
            self.vae_hidden_layer = VaeHiddenLayer(zs_weight_dim)

        if zs_weight_path == 'rand':
            self.zs_weight = nn.Parameter(zs_weight)
        else:
            self.register_buffer('zs_weight', zs_weight)

        self.num_steps_region_to_text = num_steps_region_to_text
        self.num_steps_region_to_image = num_steps_region_to_image

        if use_ddpm:
            self.ddpm_module = DiffusionMLP(zs_weight_dim, hidden_dim, zs_weight_dim, num_bottleneck_layers, num_steps_region_to_text)
            self.diffuser = GaussianDiffusion(timesteps=num_steps_region_to_text, beta_schedule='cosine')

            if self.refine_cond_type == 'att' and self.with_region_to_image:
                self.proj = nn.Linear(512,512)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'input_shape': input_shape,
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            'zs_weight_path': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
            'zs_weight_dim': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            'use_bias': cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            'norm_weight': cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT,
            'norm_temperature': cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
            'use_ddpm': cfg.MODEL.ROI_BOX_HEAD.USE_DDPM,
            'num_steps_region_to_text': cfg.DDPM.NUM_STEPS_REGION_TO_TEXT,
            'num_steps_region_to_image': cfg.DDPM.NUM_STEPS_REGION_TO_IMAGE,
            'norm_ddpm_sampling': cfg.DDPM.NORM_DDPM_SAMPLING,
            'num_bottleneck_layers': cfg.DDPM.NUM_BOTTLENECK_LAYERS,
            'hidden_dim': cfg.DDPM.HIDDEN_DIM,
            'ctx_dim': cfg.DDPM.CTX_DIM,
            'with_cond_noise': cfg.DDPM.WITH_COND_NOISE,
            'with_region_to_image': cfg.DDPM.WITH_REGION_TO_IMAGE,
            'rec_loss_weight': cfg.DDPM.REC_LOSS_WEIGHT,
            'kl_loss_weight': cfg.DDPM.KL_LOSS_WEIGHT,
            'rec_loss_l1': cfg.DDPM.REC_LOSS_L1,
            'with_middle_cond_noise': cfg.DDPM.WITH_MIDDLE_COND_NOISE,
            'refine_cond_type': cfg.DDPM.REFINE_COND_TYPE,
            'dual_forward': cfg.DDPM.DUAL_FORWARD,
            'with_over_sampling': cfg.DDPM.WITH_OVER_SAMPLING,
        }

    def sampling_prompt(self, zs_weight):
        if not self.with_prompt_dist:
            return zs_weight
        else:
            if self.training:
                indx1 = torch.randint(0, self.num_prompts, (self.num_classes,))    
                indx2 = torch.arange(self.num_classes)
                zs_weight = zs_weight[indx1, :, indx2].t() # num_classes+1, 512 -> 512, num_classes+1
                return zs_weight
            else:
                zs_weight = zs_weight.mean(0)
                return zs_weight
    def refine_cond(self, x_obj, x_img):

        if self.refine_cond_type == 'att':
            roi_cond = F.sigmoid(self.proj(x_img)) * x_obj
        elif self.refine_cond_type == 'add':
            roi_cond = 0.5 * (x_img +  x_obj)
        elif self.refine_cond_type == 'obj':
            roi_cond = x_obj
        elif self.refine_cond_type == 'img':
            roi_cond = x_img
        else:
            raise ValueError
        return roi_cond

    def forward(self, x, classifier=None):
        """
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        """
        x = self.linear(x) # num_rois, D
        zs_weight = self.zs_weight  # D C
        zs_weight = self.sampling_prompt(zs_weight)
        mu, logvar = self.vae_hidden_layer(x) if self.with_cond_noise else (None,None) 

        if self.training: 
            noise = self.vae_hidden_layer.reparameterize(mu, logvar) if self.with_cond_noise else None
            middle_noise = noise if self.with_middle_cond_noise else None
            if self.with_region_to_image:
                img_embed = self.diffuser.part_sample(self.ddpm_module, x, noise=noise, num_setps=self.num_steps_region_to_image,middle_noise=middle_noise)        
                roi_cond = self.refine_cond(x, img_embed)
            else:
                roi_cond = x
            x = self.diffuser.sample(self.ddpm_module,  roi_cond, noise=noise, middle_noise=middle_noise) 
        else:
            # noise = self.vae_hidden_layer.reparameterize(mu, logvar) if self.with_cond_noise else None
            noise = self.vae_hidden_layer.reparameterize(mu, logvar, deterministic=True) if self.with_cond_noise else None
            middle_noise = noise if self.with_middle_cond_noise else None

            if self.with_region_to_image:
                img_embed = self.diffuser.part_sample(self.ddpm_module, x, noise=noise, num_setps=self.num_steps_region_to_image, middle_noise=middle_noise)
                roi_cond = self.refine_cond(x, img_embed)
            else:
                roi_cond = x
            x = self.diffuser.sample(self.ddpm_module,  roi_cond, noise=noise, middle_noise=middle_noise) 

        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=1)
            
        x = torch.mm(x, zs_weight)

        if self.use_bias:
            x = x + self.cls_bias
        return x

    def obj_to_txt_diff(self, region_feats, label, only_fg=False):
        '''
        objec-to-text diffusion
        region_feats: condition
        text_feat: target
        '''
        loss_dict = {}
        x = self.linear(region_feats)
        zs_weight = self.sampling_prompt(self.zs_weight).t()
        if self.with_cond_noise:
            mu, logvar = self.vae_hidden_layer(x)
            loss_kl = self.vae_hidden_layer.kld_loss(mu, logvar)
            noise = self.vae_hidden_layer.reparameterize(mu, logvar)
            middle_noise = noise if self.with_middle_cond_noise else None   
            loss_dict.update({'kl_loss': loss_kl}) 
        else:
            noise = None
            middle_noise = None

        x_0 = zs_weight[label]
        if self.with_region_to_image:
            img_embed = self.diffuser.part_sample(self.ddpm_module, x, noise=noise,middle_noise=middle_noise, num_setps=self.num_steps_region_to_image)
            roi_cond = self.refine_cond(x, img_embed)
        else:
            roi_cond = x

        if self.dual_forward:
            loss_rec = self.diffuser.ddpm_forward_dual(
                self.ddpm_module,
                x_0, roi_cond,
                noise=None,
                cond_noise=noise,
                 use_l1=self.rec_loss_l1)
        elif self.with_over_sampling:
            loss_rec = self.diffuser.ddpm_forward_oversample(
                self.ddpm_module,
                x_0, 
                roi_cond, 
                noise=noise, 
                )
        else:
            loss_rec = self.diffuser.ddpm_forward(
                self.ddpm_module,
                x_0, 
                roi_cond, 
                noise=noise, 
                use_l1=self.rec_loss_l1)
        loss_dict.update({'obj_to_txt_gen_loss': loss_rec})


        return loss_dict


    def obj_to_img_diff(self, region_feats, clip_img_embed, oversampling=False):
        '''
        objec-to-image diffusion
        gerneration conditon: region_feats
        gerneration target: clip_img_embed
        '''
        x = self.linear(region_feats)    
        clip_img_embed = F.normalize(clip_img_embed, p=2, dim=1)
        if self.with_cond_noise:
            mu, logvar = self.vae_hidden_layer(x)
            noise = self.vae_hidden_layer.reparameterize(mu, logvar)
            middle_noise = noise if self.with_middle_cond_noise else None    
        else:
            noise = None
            loss_kl = None
            middle_noise= None
        x = self.diffuser.part_sample(self.ddpm_module, x, noise=noise, middle_noise=middle_noise, num_setps=self.num_steps_region_to_image)
        if self.norm_weight:
            x =  F.normalize(x, p=2, dim=1)
        loss_cons =  F.l1_loss(x, clip_img_embed)
        return loss_cons

class WeightTransferZeroShotClassifier(nn.Module):
    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            num_classes: int,
            zs_weight_path: str,
            zs_weight_dim: int = 512,
            use_bias: float = 0.0,
            norm_weight: bool = True,
            norm_temperature: float = 50.0,
            use_ddpm: bool = False,
            num_steps_region_to_text: int = 1000,
            norm_ddpm_sampling: str = 'clamp',
            num_bottleneck_layers: int = 2,
            hidden_dim: int = 256,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature

        self.use_ddpm = use_ddpm
        self.norm_ddpm_sampling = norm_ddpm_sampling
        if use_ddpm:
            self.ddpm_module = DiffusionMLP(zs_weight_dim, hidden_dim, zs_weight_dim, num_bottleneck_layers, num_steps_region_to_text)
            self.diffuser = GaussianDiffusion(timesteps=num_steps_region_to_text, beta_schedule='cosine')

        self.use_bias = use_bias < 0
        # if self.use_bias:
        #     self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        # this layer now acts as frozen distilled linear layer
        self.linear = nn.Linear(input_size, zs_weight_dim)
        for param in self.linear.parameters():
            param.requires_grad = False

        # FC weight transfer layers
        self.fc1 = nn.Linear(input_size, zs_weight_dim)
        self.fc2 = nn.Linear(zs_weight_dim, input_size)
        self.relu = nn.LeakyReLU(0.1)
        # FC residual layers
        self.fc3 = nn.Linear(input_size, 1024)
        self.fc4 = nn.Linear(1024, zs_weight_dim)

        if zs_weight_path == 'rand':
            zs_weight = torch.randn((zs_weight_dim, num_classes))
            nn.init.normal_(zs_weight, std=0.01)
        else:
            zs_weight = torch.tensor(
                np.load(zs_weight_path),
                dtype=torch.float32).permute(1, 0).contiguous()  # D x C
        # zs_weight = torch.cat(
        #     [zs_weight, zs_weight.new_zeros((zs_weight_dim, 1))],
        #     dim=1)  # D x (C + 1)

        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=0)

        if zs_weight_path == 'rand':
            self.zs_weight = nn.Parameter(zs_weight)
        else:
            self.register_buffer('zs_weight', zs_weight)

        assert self.zs_weight.shape[1] == num_classes + 1, self.zs_weight.shape

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'input_shape': input_shape,
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            'zs_weight_path': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
            'zs_weight_dim': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            'use_bias': cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            'norm_weight': cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT,
            'norm_temperature': cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
            'use_ddpm': cfg.MODEL.ROI_BOX_HEAD.USE_DDPM,
            'num_steps_region_to_text': cfg.DDPM.NUM_DDPM_SETPS,
            'norm_ddpm_sampling': cfg.DDPM.NORM_DDPM_SAMPLING,
            'num_bottleneck_layers': cfg.DDPM.NUM_BOTTLENECK_LAYERS,
            'hidden_dim': cfg.DDPM.HIDDEN_DIM,
        }

    def _forward_middle(self, x):
        # Compute the weights through transfer function
        t = self.fc1(self.linear.weight) # run2
        t_act = self.relu(t)
        transfer_weights = self.fc2(t_act)
        # Pass though linear layer after weight transfer
        res_x = self.fc3(x)
        res_x = self.relu(res_x)
        res_x = self.fc4(res_x)

        x = res_x + F.linear(x, weight=transfer_weights)
        return x

    def forward(self, x, classifier=None):
        """
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        """
        x = self._forward_middle(x)

        if self.use_ddpm:
            x = self.diffuser.sample(self.ddpm_module, x, clip_denoised=self.norm_ddpm_sampling) # clamp
                    

        if classifier is not None:
            zs_weight = classifier.permute(1, 0).contiguous()  # D x C'
            zs_weight = F.normalize(zs_weight, p=2, dim=0) \
                if self.norm_weight else zs_weight
        else:
            zs_weight = self.zs_weight
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=1)
        x = torch.mm(x, zs_weight)
        if self.use_bias:
            x = x + self.cls_bias
        return x

    def get_noise_reconstruction_loss(self, x, label, only_fg=False):
        # fg_mask = label != 80
        x_0 = self.zs_weight.t()[label]

        cond = self._forward_middle(x)
        loss = self.diffuser.ddpm_forward_oversample(self.ddpm_module, x_0, cond)
        return loss
