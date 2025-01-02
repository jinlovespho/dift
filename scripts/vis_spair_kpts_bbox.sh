CUDA_VISIBLE_DEVICES=0 python vis_spair_kpts_bbox.py \
                                    --dataset_path ./data/SPair-71k \
                                    --save_path ./feats/spair_ft \
                                    --dift_model sd \
                                    --img_size 768 768 \
                                    --t 261 \
                                    --up_ft_index 1 \
                                    --ensemble_size 8