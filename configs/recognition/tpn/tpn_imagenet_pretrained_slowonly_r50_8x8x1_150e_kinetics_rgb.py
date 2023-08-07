_base_ = [
    '../../_base_/models/tpn_slowonly_r50.py',
    '../../_base_/default_runtime.py'
]

# dataset_type = 'RawframeDataset'
# #data_root = '/home/dataset/video/VIDEO/K400/images'                 #old version
# data_root = '/home/dataset/video/VIDEO/K400/new_256_frames/train'
# #data_root_val = '/home/dataset/video/VIDEO/K400/images'            #old version
# data_root_val = '/home/dataset/video/VIDEO/K400/new_256_frames/val'
# # ann_file_train = 'data/kinetics400/kinetics400_train_list_rawframes.txt' #old version
# # ann_file_val = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
# # ann_file_test = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
ann_file_train = 'data/kinetics400/train_for_snip_24frm.txt'
ann_file_val = 'data/kinetics400/val_for_snip_24frm.txt'
ann_file_test = 'data/kinetics400/val_for_snip_24frm.txt'

#---------------------------------
#hqy 10.16 disk-testing
# dataset_type = 'RawframeDataset'
# data_root = '/media/ubuntu/data/gzl/train_frames_tmp'  #no.1
# # data_root = '/media/ubuntu/PSSD/gzl/tmp_frames_train'  #no.2
# # data_root_val = '/media/ubuntu/data/gzl/train_frames_tmp' #no.1
# data_root_val = '/media/ubuntu/PSSD/gzl/tmp_frames_train'  #no.2
# # # ann_file_train = 'data/kinetics400/kinetics400_train_list_rawframes.txt'
# # # ann_file_val = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
# # # ann_file_test = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
# ann_file_train = '/media/ubuntu/data/gzl/train_tmp.txt'
# ann_file_val = '/media/ubuntu/data/gzl/train_tmp.txt'
# ann_file_test = '/media/ubuntu/data/gzl/train_tmp.txt'
#----------------------------------


#mini-K200-dataHDD
dataset_type = 'RawframeDataset'
data_root = '/media/ubuntu/data/HDDdata/K400_frame_new/train'
data_root_val = '/media/ubuntu/data/HDDdata/K400_frame_new/val'
# ann_file_train = '/home/dataset/video/VIDEO/mini-K200/train_K200.txt' #'/media/gzl/nvme/K400_frame/K400_txt/train.txt'
# ann_file_val = '/home/dataset/video/VIDEO/mini-K200/val_K200.txt' #'/media/gzl/nvme/K400_frame/K400_txt/val_wo_mp4.txt'
# ann_file_test = '/home/dataset/video/VIDEO/mini-K200/val_K200.txt' # '/media/gzl/nvme/K400_frame/K400_txt/val_wo_mp4.txt'

#--------------------
#syncBN  10.12
# model = dict(
#     backbone=dict(
#         norm_cfg=dict(type='SyncBN', requires_grad=True)))
#-------------------

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
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
        clip_len=8,
        frame_interval=8,
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
        clip_len=8,
        frame_interval=8,
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
    videos_per_gpu=8,
    #workers_per_gpu=6,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001,
    nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[75, 125])
total_epochs = 150

# runtime settings
work_dir = './work_dirs/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics400_rgb'  # noqa: E501
