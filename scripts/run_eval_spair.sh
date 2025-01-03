CUDA_VISIBLE_DEVICES=0 python eval_spair.py \
                                    --dataset_path ./data/SPair-71k \
                                    --save_path ./extracted_feats/spair_dift \
                                    --dift_model sd \
                                    --img_size 768 768 \
                                    --t 261 \
                                    --up_ft_index 1 \
                                    --ensemble_size 8 \
                                    --save_vis_pred_kpts_dir './vis/spair/dift' \
                                    --vis_pred_kpts \


# main path: /media/dataset1/jinlovespho/github/icml2025/dift
# dataset_path: ./data/SPair-71k
# save_path: ./extracted_feats/spair_dift
# dift_model: sd
# img_size: [768, 768]
# t: 261
# up_ft_index: 1
# ensemble_size: 8
# vis_pred_kpts: True
# save_vis_pred_kpts_dir: ./vis/spair/dift
# saving all test images' features...
# 100%|████████████████████████████████████████████████████████████████████████████| 18/18 [15:58<00:00, 53.28s/it]
# Evaluating for category ==> train
# 100%|███████████████████████████████████████████████████████████████████████████| 756/756 [00:53<00:00, 14.14it/s]
# train per image PCK@0.1: 67.49
# train per point PCK@0.1: 70.96
# Evaluating for category ==> boat
# 100%|███████████████████████████████████████████████████████████████████████████| 702/702 [00:22<00:00, 31.31it/s]
# boat per image PCK@0.1: 30.65
# boat per point PCK@0.1: 34.29
# Evaluating for category ==> cow
# 100%|███████████████████████████████████████████████████████████████████████████| 640/640 [00:34<00:00, 18.55it/s]
# cow per image PCK@0.1: 71.02
# cow per point PCK@0.1: 76.35
# Evaluating for category ==> cat
# 100%|███████████████████████████████████████████████████████████████████████████| 600/600 [00:39<00:00, 15.32it/s]
# cat per image PCK@0.1: 78.04
# cat per point PCK@0.1: 77.54
# Evaluating for category ==> car
# 100%|███████████████████████████████████████████████████████████████████████████| 564/564 [00:21<00:00, 26.43it/s]
# car per image PCK@0.1: 33.73
# car per point PCK@0.1: 48.36
# Evaluating for category ==> pottedplant
# 100%|███████████████████████████████████████████████████████████████████████████| 862/862 [00:29<00:00, 28.76it/s]
# pottedplant per image PCK@0.1: 52.58
# pottedplant per point PCK@0.1: 57.48
# Evaluating for category ==> bottle
# 100%|███████████████████████████████████████████████████████████████████████████| 870/870 [00:38<00:00, 22.54it/s]
# bottle per image PCK@0.1: 44.92
# bottle per point PCK@0.1: 46.40
# Evaluating for category ==> chair
# 100%|███████████████████████████████████████████████████████████████████████████| 646/646 [00:23<00:00, 27.16it/s]
# chair per image PCK@0.1: 34.49
# chair per point PCK@0.1: 39.24
# Evaluating for category ==> bus
# 100%|███████████████████████████████████████████████████████████████████████████| 644/644 [00:28<00:00, 22.45it/s]
# bus per image PCK@0.1: 39.46
# bus per point PCK@0.1: 52.28
# Evaluating for category ==> sheep
# 100%|███████████████████████████████████████████████████████████████████████████| 664/664 [00:25<00:00, 26.27it/s]
# sheep per image PCK@0.1: 45.53
# sheep per point PCK@0.1: 56.55
# Evaluating for category ==> motorbike
# 100%|███████████████████████████████████████████████████████████████████████████| 702/702 [00:23<00:00, 30.19it/s]
# motorbike per image PCK@0.1: 49.86
# motorbike per point PCK@0.1: 52.88
# Evaluating for category ==> bicycle
# 100%|███████████████████████████████████████████████████████████████████████████████████████████| 650/650 [00:25<00:00, 25.97it/s]
# bicycle per image PCK@0.1: 52.87
# bicycle per point PCK@0.1: 54.53
# Evaluating for category ==> horse
# 100%|███████████████████████████████████████████████████████████████████████████████████████████| 600/600 [00:27<00:00, 21.54it/s]
# horse per image PCK@0.1: 57.36
# horse per point PCK@0.1: 61.18
# Evaluating for category ==> tvmonitor
# 100%|███████████████████████████████████████████████████████████████████████████████████████████| 692/692 [00:45<00:00, 15.16it/s]
# tvmonitor per image PCK@0.1: 60.07
# tvmonitor per point PCK@0.1: 63.65
# Evaluating for category ==> dog
# 100%|███████████████████████████████████████████████████████████████████████████████████████████| 600/600 [00:31<00:00, 19.34it/s]
# dog per image PCK@0.1: 51.10
# dog per point PCK@0.1: 54.69
# Evaluating for category ==> bird
# 100%|███████████████████████████████████████████████████████████████████████████████████████████| 702/702 [00:29<00:00, 24.16it/s]
# bird per image PCK@0.1: 78.38
# bird per point PCK@0.1: 80.15
# Evaluating for category ==> person
# 100%|███████████████████████████████████████████████████████████████████████████████████████████| 650/650 [00:28<00:00, 23.03it/s]
# person per image PCK@0.1: 40.93
# person per point PCK@0.1: 45.86
# Evaluating for category ==> aeroplane
# 100%|███████████████████████████████████████████████████████████████████████████████████████████| 690/690 [00:34<00:00, 20.17it/s]
# aeroplane per image PCK@0.1: 61.83
# aeroplane per point PCK@0.1: 63.56
# All per image PCK@0.1: 52.84
# All per point PCK@0.1: 59.41