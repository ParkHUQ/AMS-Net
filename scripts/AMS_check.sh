# for e in $(seq 104 -1 104)
# do
# echo epoch:$e
./tools/dist_test.sh configs/recognition/ams/ams_r50_1x1x8_110e_sthv1_rgb.py models/SomethingV1/ams_sthv1_8fx1x1.pth  8 --eval top_k_accuracy mean_class_accuracy --average-clips prob
# done
