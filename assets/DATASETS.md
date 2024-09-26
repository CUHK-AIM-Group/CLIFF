# Prepare datasets for Open Vocabulary Detection

## COCO Open Vocabulary

This is consistent to [OCD](https://github.com/hanoonaR/object-centric-ovd).
 
The annotations for OVD training `instances_train2017_seen_2_oriorder.json`, `instances_train2017_seen_2_oriorder_cat_info.json`
and evaluation `instances_val2017_all_2_oriorder.json`, and annotations for image-level supervision `captions_train2017_tags_allcaps_pis.json` 
can be downloaded from [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EhJUv1cVKJtCnZJzsvGgVwYBgxP6M9TWsD-PBb_KgOjhmQ?e=iYkfDZ).

The CLIP image features on class-agnostic MAVL proposals for region-based knowledge distillation (RKD) and
class-specific proposals for pseudo image-level supervision (PIS) can be downloaded from [here](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EeJuo844j8FIsnuiX3wBxCgBcBR2MSjbhiLCuA4OC2cSWg?e=5BeESO).
Untar the file `coco_props.tar.gz` and place in the corresponding location as shown below:

```
coco/
    train2017/
    val2017/
    annotations/
        captions_train2017.json
        instances_train2017.json 
        instances_val2017.json
        captions_train2017_tags_allcaps_pis.json
    zero-shot/
        instances_train2017_seen_2_oriorder.json
        instances_val2017_all_2_oriorder.json
        instances_train2017_seen_2_oriorder_cat_info.json
MAVL_proposals
    coco_props/
        classagnostic_distilfeats/
            000000581921.pkl
            000000581929.pkl
            ....
        class_specific/
            000000581921.pkl
            000000581929.pkl
            ....
```

Otherwise, follow the following instructions to generate them from the COCO standard annotations. We follow the 
code-base of [Detic](https://github.com/facebookresearch/Detic) for the dataset preperation.

1) COCO annotations

Following the work of [OVR-CNN](https://github.com/alirezazareian/ovr-cnn/blob/master/ipynb/003.ipynb), we first
create the open-vocabulary COCO split. The converted files should be placed as shown below,
```
coco/
    zero-shot/
        instances_train2017_seen_2.json
        instances_val2017_all_2.json
```
These annotations are then pre-processed for easier evaluation, using the following commands:

```
python tools/get_coco_zeroshot_oriorder.py --data_path datasets/coco/zero-shot/instances_train2017_seen_2.json
python tools/get_coco_zeroshot_oriorder.py --data_path datasets/coco/zero-shot/instances_val2017_all_2.json
```

2) Annotations for Image-level labels from COCO-Captions

The final annotation to prepare is `captions_train2017_tags_allcaps_pis.json`.
For the Image-level supervision, we use the COCO-captions annotations and filter them with the MAVL predictions.
First generate the Image-level labels from COCO-captions with the command, 
```
python tools/get_cc_tags.py --cc_ann datasets/coco/annotations/captions_train2017.json 
    --out_path datasets/coco/annotations/captions_train2017_tags_allcaps.json  
    --allcaps --convert_caption --cat_path datasets/coco/annotations/instances_val2017.json
```
This creates `datasets/coco/captions_train2017_tags_allcaps.json`.

To ignore the remaining classes from the COCO that are not included in seen(65)+ unseen(17),
`instances_train2017_seen_2_oriorder_cat_info.json` is used by the flag `IGNORE_ZERO_CATS`.  This is created by 
```
python tools/get_lvis_cat_info.py --ann datasets/coco/zero-shot/instances_train2017_seen_2_oriorder.json
```

3) Proposals for PIS

With the Image-level labels from COCO-captions `captions_train2017_tags_allcaps_pis.json`, 
generate pseudo-proposals for PIS using MAVL with the below command. Download the checkpoints from the external submodule.
```
python tools/get_ils_labels.py -ckpt <mavl pretrained weights path> -dataset coco -dataset_dir datasets/coco
        -output datasets/MAVL_proposals/coco_props/class_specific
```
The class-specific pseudo-proposals will be stored as individual pickle files for each image.

Filter the Image-level annotation `captions_train2017_tags_allcaps.json` from COCO-captions with the pseudo-proposals
from MAVL with the command,
```
python tools/update_cc_pis_annotations.py --ils_path datasets/MAVL_proposals/coco_props/class_specific 
    --cc_ann_path datasets/coco/annotations/captions_train2017_tags_allcaps.json 
    --output_ann_path datasets/coco/annotations/captions_train2017_tags_allcaps_pis.json
```
This creates `datasets/coco/captions_train2017_tags_allcaps_pis.json`.

4) Regions and CLIP embeddings for RKD

Generate the class-agnostic proposals for the images in the training set. The class-agnostic pseudo-proposals and 
corresponding CLIP images features will be stored as individual pickle files for each image.
Note: It does not depend on  steps 2 and 3.
```
python tools/get_rkd_clip_feat.py -ckpt <mavl pretrained weights path> -dataset coco -dataset_dir datasets/coco
        -output datasets/MAVL_proposals/coco_props/classagnostic_distilfeats
```
