# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetTSN_Temporal',
        pretrained='torchvision://resnet50',
        depth=50,
        out_indices=(3, ),
        temporal_block_indices=(0, 2),
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=174,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
