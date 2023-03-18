# learn how to use mmselfsup

# ------------------------ #
# MAE (demo)
# ------------------------ #
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/selfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k.py


# ------------------------ #
# SimMIM 
# ------------------------ #
# base (demo from the document)
python tools/train.py configs/selfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k.py

# tiny (modified by me)
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/selfsup/simmim/simmim_swin-tiny_8xb256-amp-coslr-100e_in1k-192.py
