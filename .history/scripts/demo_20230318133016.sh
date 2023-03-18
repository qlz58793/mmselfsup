# learn how to use mmselfsup

# ------------------------ #
# MAE (demo)
# ------------------------ #
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/selfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k.py


# ------------------------ #
# SimMIM 
# ------------------------ #
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/selfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k.py
