#Trained on 4 x Titan RTX GPUs
./tools/dist_train.sh configs/recognition/ams/ss_ams_r50_1x1x16_120e_sthv2_rgb.py 4  --work-dir work_dirs/ss_ams_r50_1x1x16_120e_sthv2_rgb --seed 0 --deterministic
