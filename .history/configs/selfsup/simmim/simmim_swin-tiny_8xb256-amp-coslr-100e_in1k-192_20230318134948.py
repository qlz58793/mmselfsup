_base_ = 'simmim_swin-base_16xb128-amp-coslr-100e_in1k-192.py'

# dataset 8 GPUs x 256
train_dataloader = dict(batch_size=256, num_workers=16)

model = dict(
    backbone=dict(
        arch='T',),
    neck=dict(type='SimMIMNeck', in_channels=192 * 2**3, encoder_stride=32),
    head=dict(
        type='SimMIMHead',
        patch_size=4,
        loss=dict(type='SimMIMReconstructionLoss', encoder_in_channels=3)))
