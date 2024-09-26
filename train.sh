source ~/miniconda3/etc/profile.d/conda.sh

conda activate cliff

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net_diffusion.py --num-gpus 4 \
--dist-url tcp://127.0.0.1:50225 \
--config-file configs/CLIFF_COCO_RCNN-C4_obj2txt_stage1.yaml 

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net_diffusion.py --num-gpus 4 \
--dist-url tcp://127.0.0.1:50225 \
--config-file configs/CLIFF_COCO_RCNN-C4_obj2img2txt_stage2.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net_diffusion.py --num-gpus 4 \
--dist-url tcp://127.0.0.1:50225 \
--config-file configs/CLIFF_COCO_RCNN-C4_obj2img2txt_final.yaml