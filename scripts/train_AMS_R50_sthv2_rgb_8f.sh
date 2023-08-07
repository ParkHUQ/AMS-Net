#Trained on 8 x RTX 2080Ti GPUs
./tools/dist_train.sh configs/recognition/ams/ams_r50_1x1x8_120e_sthv2_rgb.py 8  --work-dir work_dirs/ams_r50_1x1x8_120e_sthv2_rgb --seed 0 --deterministic
