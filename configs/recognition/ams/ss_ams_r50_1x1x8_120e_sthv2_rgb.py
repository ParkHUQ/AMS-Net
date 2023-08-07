_base_ = [
    '../../_base_/models/ss_ams_r50.py', '../../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/media/ubuntu/data/HDDdata/STHV2/20bn-something-something-v2-frames'
data_root_val = '/media/ubuntu/data/HDDdata/STHV2/20bn-something-something-v2-frames'
ann_file_train = 'data/somethingv2/train_videofolder.txt'
ann_file_val = 'data/somethingv2/val_videofolder.txt'
#ann_file_test = 'data/somethingv2/test_videofolder.txt'
ann_file_test = 'data/somethingv2/val_videofolder.txt'

model = dict(backbone=dict(num_segments=8, gamma=1),
             neck=dict(gamma=1, temporal_modulation_cfg=dict(downsample_scales=(8, 8))),
             test_cfg=dict(average_clips='prob', fcn_test=True))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SnippetSampleFrames', clip_len=1, frame_interval=1, num_clips=8, new_length=3),
    dict(type='RawFrameDecode'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter', color_space_aug=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SnippetSampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        new_length=3,
        shift_val=3,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SnippetSampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        new_length=3,
        shift_val=3,
        twice_sample=True,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=6,
    train_dataloader=dict(drop_last=True),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl='{:06}.jpg',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl='{:06}.jpg',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        filename_tmpl='{:06}.jpg',
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD', lr=0.008, momentum=0.9, weight_decay=0.0005,
    nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[75, 105])
total_epochs = 120

# runtime settings
work_dir = './work_dirs/ss_ams_r50_1x1x8_120e_sthv2_rgb'


