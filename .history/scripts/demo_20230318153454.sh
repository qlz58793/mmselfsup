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

# base (fine-tune 20 epochs and initialize with the pre-trained model of 100 epochs)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh \
    configs/selfsup/simmim/simmim_swin-base_4xb128-amp-coslr-ft20e_in1k-192.py \
    4 

# tiny (modified by me)
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/selfsup/simmim/simmim_swin-tiny_8xb256-amp-coslr-100e_in1k-192.py

# tiny / multi-gpu
# bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} [optional arguments]
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh \
    configs/selfsup/simmim/simmim_swin-tiny_8xb256-amp-coslr-100e_in1k-192.py \
    4 

