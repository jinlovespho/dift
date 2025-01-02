CUDA_VISIBLE_DEVICES=0 python extract_dift.py \
                                        --model_id stabilityai/stable-diffusion-2-1 \
                                        --input_path ./assets/cat.png \
                                        --output_path ./feats/dift_cat.pt \
                                        --img_size 768 768 \
                                        --t 261 \
                                        --up_ft_index 1 \
                                        --prompt 'a photo of a cat' \
                                        --ensemble_size 8

