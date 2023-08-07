_base_ = [
    '../../_base_/models/tpn_slowonly_r50.py',
    '../../_base_/default_runtime.py'
]

dataset_type = 'RawframeDataset'
data_root = '/media/ubuntu/data/HDDdata/FineGym/video2ele_frm/annotation_v1.1/final_pipeline/data/FineGym-frames'
data_root_val = '/media/ubuntu/data/HDDdata/FineGym/video2ele_frm/annotation_v1.1/final_pipeline/data/FineGym-frames'
ann_file_train = '/media/ubuntu/data/HDDdata/FineGym/annotations/gym288_train_frame_v1.txt'
ann_file_val = '/media/ubuntu/data/HDDdata/FineGym/annotations/gym288_val_frame.txt'
ann_file_test = '/media/ubuntu/data/HDDdata/FineGym/annotations/gym288_val_frame.txt'

#--------------------
#syncBN  10.12
# model = dict(
#     backbone=dict(
#         norm_cfg=dict(type='SyncBN', requires_grad=True)))
#-------------------

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=4, frame_interval=16, num_clips=1),
    #dict(type='FrameSelector'),
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
        clip_len=4,
        frame_interval=16,
        num_clips=1,
        test_mode=True),
    #dict(type='FrameSelector'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    #dict(type='ColorJitter', color_space_aug=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=4,
        frame_interval=16,
        num_clips=10,
        test_mode=True),
    #dict(type='FrameSelector'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=20,
    workers_per_gpu=6,
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
        pipeline=val_pipeline))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0001,
    nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[90, 110])
total_epochs = 120

# runtime settings
work_dir = './work_dirs/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics400_rgb'  # noqa: E501
