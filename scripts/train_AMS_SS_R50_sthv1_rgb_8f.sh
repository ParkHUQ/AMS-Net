#Trained on 8 x RTX 2080Ti GPUs
./tools/dist_train.sh configs/recognition/ams/ss_ams_r50_1x1x8_110e_sthv1_rgb.py 8  --work-dir work_dirs/ss_ams_r50_1x1x8_110e_sthv1_rgb --seed 0 --deterministic
