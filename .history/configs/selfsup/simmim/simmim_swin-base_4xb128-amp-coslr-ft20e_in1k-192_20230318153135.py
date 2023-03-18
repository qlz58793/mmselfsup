_base_ = 'simmim_swin-base_16xb128-amp-coslr-100e_in1k-192.py'

# dataset 4 GPUs x 128
train_dataloader = dict(batch_size=128, num_workers=16)

# optimizer wrapper
optimizer = dict(
#     type='AdamW', lr=2e-4 * 2048 / 512, betas=(0.9, 0.999), eps=1e-8)
    type='AdamW', lr=5e-5, betas=(0.9, 0.999), eps=1e-8)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=5.0),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        # start_factor=1e-6 / 2e-4,
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=15,
        # eta_min=1e-5 * 2048 / 512,
        eta_min=1e-5,
        by_epoch=True,
        begin=5,
        end=20,
        convert_to_iter_based=True)
]

# schedule
train_cfg = dict(max_epochs=20)

# runtime
default_hooks = dict(logger=dict(type='LoggerHook', interval=100))
