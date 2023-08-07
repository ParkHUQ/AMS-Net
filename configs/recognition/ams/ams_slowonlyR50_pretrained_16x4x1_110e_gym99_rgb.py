_base_ = [
    '../../_base_/models/atpn_slowonly_r50.py',
    '../../_base_/default_runtime.py'
]

#data settings
dataset_type = 'RawframeDataset'
data_root = '/media/ubuntu/data/HDDdata/FineGym/video2ele_frm/annotation_v1.1/final_pipeline/data/FineGym-frames'
data_root_val = '/media/ubuntu/data/HDDdata/FineGym/video2ele_frm/annotation_v1.1/final_pipeline/data/FineGym-frames'
ann_file_train = 'data/finegym/gym99_train_frame_v1.txt'
ann_file_val = 'data/finegym/gym99_val_frame.txt'
ann_file_test = 'data/finegym/gym99_val_frame.txt'


model=dict(
    backbone=dict(pretrained=None),
    neck=dict(temporal_modulation_cfg=dict(downsample_scales=(16, 16)),
              aux_head_cfg=dict(out_channels=99, loss_weight=0.5)),
    cls_head=dict(num_classes=99))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter', color_space_aug=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=5,
    workers_per_gpu=1,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl='img_{:05d}.jpg',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl='img_{:05d}.jpg',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        filename_tmpl='img_{:05d}.jpg',
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001, #gzl 10.06
    nesterov=True,
    paramwise_cfg=dict(custom_keys={'neck.aux_head.fc': dict(lr_mult=5, decay_mult=0), 'head.fc_cls': dict(lr_mult=5, decay_mult=0)}))
#)
# this lr is used for 8 gpus, bs = 40 x 2 (gradient cumulative)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policyss
lr_config = dict(policy='step', step=[65, 85])
total_epochs = 110

# runtime settings
work_dir = './work_dirs/ams_slowonlyR50_pretrained_16x4x1_110e_gym99_rgb' 
load_from = '/home/zhangli/Project/AMG-slow-v1.a.2_cp/models/idg5_lr1e-2_step75100_epoch_105.pth'