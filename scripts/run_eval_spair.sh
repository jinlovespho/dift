CUDA_VISIBLE_DEVICES=3 python eval_spair.py \
                                    --dataset_path ./data/SPair-71k \
                                    --save_path ./extracted_feats/spair_dift \
                                    --dift_model sd \
                                    --is_feat_extracted True \
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
# /home/cvlab05/anaconda3/envs/pho_dift/lib/python3.10/site-packages/huggingface_hub/file_download.py:795: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to
#  force a new download, use `force_download=True`.                                                                                                                                                                                                  
#   warnings.warn(                                                                                                                                                                                                                                   
# saving all test images' features...                                                                                                                                                                                                                
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 18/18 [14:00<00:00, 46.67s/it]                                                                                                    
# /media/dataset1/jinlovespho/github/icml2025/dift/eval_spair.py:78: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct ma
# licious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped
#  to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals
# `. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.                                   
#   output_dict = torch.load(os.path.join(args.save_path, f'{cat}.pth'))                                                                                                                                                                             
# Evaluating for category ==> train                                                                                                                                                                                                                  
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 756/756 [00:50<00:00, 14.90it/s]                                                                                                    
# train per image PCK@0.1: 68.33                                                                                                                                                                                                                     
# train per point PCK@0.1: 71.45                                                                                                                                                                                                                     
# Evaluating for category ==> boat                                                                                                                                                                                                                   
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 702/702 [00:25<00:00, 27.94it/s]                                                                                                    
# boat per image PCK@0.1: 31.17                                                                                                                                                                                                                      
# boat per point PCK@0.1: 34.55                                                                                                                                                                                                                      
# Evaluating for category ==> cow                                                                                                                                                                                                                    
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 640/640 [00:35<00:00, 18.08it/s]                                                                                                    
# cow per image PCK@0.1: 70.88                                                                                                                                                                                                                       
# cow per point PCK@0.1: 76.35                                                                                                                                                                                                                       
# Evaluating for category ==> cat                                                                                                                                                                                                                    
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [00:37<00:00, 15.80it/s]                                                                                                    
# cat per image PCK@0.1: 76.97                                                                                                                                                                                                                       
# cat per point PCK@0.1: 76.40                                                                                                                                                                                                                       
# Evaluating for category ==> car                                                                                                                
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 564/564 [00:23<00:00, 24.23it/s]                                                                                             [16/57]
# car per image PCK@0.1: 33.21
# car per point PCK@0.1: 47.83
# Evaluating for category ==> pottedplant
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 862/862 [00:33<00:00, 25.94it/s]
# pottedplant per image PCK@0.1: 51.18
# pottedplant per point PCK@0.1: 56.11
# Evaluating for category ==> bottle
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 870/870 [00:39<00:00, 21.80it/s]
# bottle per image PCK@0.1: 44.67
# bottle per point PCK@0.1: 46.06
# Evaluating for category ==> chair
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 646/646 [00:25<00:00, 25.22it/s]
# chair per image PCK@0.1: 34.35
# chair per point PCK@0.1: 38.86
# Evaluating for category ==> bus
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 644/644 [00:30<00:00, 21.09it/s]
# bus per image PCK@0.1: 39.84
# bus per point PCK@0.1: 52.57
# Evaluating for category ==> sheep
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 664/664 [00:26<00:00, 24.76it/s]
# sheep per image PCK@0.1: 45.84
# sheep per point PCK@0.1: 57.03
# Evaluating for category ==> motorbike
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 702/702 [00:25<00:00, 27.21it/s]
# motorbike per image PCK@0.1: 49.71
# motorbike per point PCK@0.1: 52.70
# Evaluating for category ==> bicycle
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 650/650 [00:27<00:00, 23.94it/s]
# bicycle per image PCK@0.1: 53.17
# bicycle per point PCK@0.1: 54.63
# Evaluating for category ==> horse
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [00:28<00:00, 21.32it/s]
# horse per image PCK@0.1: 58.00
# horse per point PCK@0.1: 61.88
# Evaluating for category ==> tvmonitor
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 692/692 [00:44<00:00, 15.50it/s]
# tvmonitor per image PCK@0.1: 59.77
# tvmonitor per point PCK@0.1: 63.33
# Evaluating for category ==> dog
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [00:31<00:00, 19.14it/s]
# dog per image PCK@0.1: 51.37
# dog per point PCK@0.1: 54.98
# Evaluating for category ==> bird
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 702/702 [00:30<00:00, 22.93it/s]
# bird per image PCK@0.1: 78.62
# bird per point PCK@0.1: 80.47
# Evaluating for category ==> person
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 650/650 [00:29<00:00, 22.13it/s]
# person per image PCK@0.1: 41.09
# person per point PCK@0.1: 46.04
# Evaluating for category ==> aeroplane
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 690/690 [00:32<00:00, 21.13it/s]
# aeroplane per image PCK@0.1: 62.01
# aeroplane per point PCK@0.1: 63.78
# All per image PCK@0.1: 52.82
# All per point PCK@0.1: 59.36