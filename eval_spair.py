import argparse
import torch
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from src.models.dift_sd import SDFeaturizer4Eval
from src.models.dift_adm import ADMFeaturizer4Eval
import os
import json
from PIL import Image
import torch.nn as nn
import cv2 


def main(args):
    for arg in vars(args):
        value = getattr(args,arg)
        if value is not None:
            print('%s: %s' % (str(arg),str(value)))

    torch.cuda.set_device(0)

    dataset_path = args.dataset_path
    test_path = 'PairAnnotation/test'
    json_list = os.listdir(os.path.join(dataset_path, test_path))
    all_cats = os.listdir(os.path.join(dataset_path, 'JPEGImages'))
    cat2json = {}

    for cat in all_cats:
        cat_list = []
        for i in json_list:
            if cat in i:
                cat_list.append(i)
        cat2json[cat] = cat_list

    # get test image path for all cats
    cat2img = {}
    for cat in all_cats:
        cat2img[cat] = []
        cat_list = cat2json[cat]
        for json_path in cat_list:
            with open(os.path.join(dataset_path, test_path, json_path)) as temp_f:
                data = json.load(temp_f)
                temp_f.close()
            src_imname = data['src_imname']
            trg_imname = data['trg_imname']
            if src_imname not in cat2img[cat]:
                cat2img[cat].append(src_imname)
            if trg_imname not in cat2img[cat]:
                cat2img[cat].append(trg_imname)

    if args.dift_model == 'sd':
        dift = SDFeaturizer4Eval(cat_list=all_cats)
    elif args.dift_model == 'adm':
        dift = ADMFeaturizer4Eval()

    ## FIRST EXTRACT FEATURES FROM THE DIFFUSION MODEL
    # print("saving all test images' features...")
    # os.makedirs(args.save_path, exist_ok=True)
    # for cat in tqdm(all_cats):  # 각 category 별로 prompt를 주기 위해 category for문을 만들었네
    #     output_dict = {}
    #     image_list = cat2img[cat]
    #     for image_path in image_list:
    #         img = Image.open(os.path.join(dataset_path, 'JPEGImages', cat, image_path))
    #         output_dict[image_path] = dift.forward(img,
    #                                             category=cat,
    #                                             img_size=args.img_size,
    #                                             t=args.t,
    #                                             up_ft_index=args.up_ft_index,
    #                                             ensemble_size=args.ensemble_size)
        
    #     print(f'Category: {cat}, number of imgs in {cat}: {len(output_dict)}')
    #     torch.save(output_dict, os.path.join(args.save_path, f'{cat}.pth'))


    total_pck = []
    all_correct = 0
    all_total = 0

    for cat in all_cats:
        cat_list = cat2json[cat]
        output_dict = torch.load(os.path.join(args.save_path, f'{cat}.pth'))

        cat_pck = []
        cat_correct = 0
        cat_total = 0

        for json_path in tqdm(cat_list):

            with open(os.path.join(dataset_path, test_path, json_path)) as temp_f:
                data = json.load(temp_f)

            SAVE_PATH = './vis/vis_matches/'
            if not os.path.exists(SAVE_PATH):
                    os.makedirs(SAVE_PATH)
                    
            VISUALIZE=True
            ## PHO_VISUALIZE BBOX and KEYPOINTS IN TARGET AND SOURCE
            if VISUALIZE:
                # Load source and target images
                src_img = cv2.imread(os.path.join(dataset_path, 'JPEGImages', cat, data['src_imname']))
                trg_img = cv2.imread(os.path.join(dataset_path, 'JPEGImages', cat, data['trg_imname']))
                # Draw source bounding box and keypoints
                src_vis = src_img.copy()
                src_bbox = data['src_bndbox']
                cv2.rectangle(src_vis, (src_bbox[0], src_bbox[1]), (src_bbox[2], src_bbox[3]), (0,0,255), 2)
                # Add bbox coordinates
                cv2.putText(src_vis, f'({src_bbox[0]},{src_bbox[1]})', (src_bbox[0], src_bbox[1]-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                cv2.putText(src_vis, f'({src_bbox[2]},{src_bbox[3]})', (src_bbox[2], src_bbox[3]+15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                # Add keypoint coordinates
                for kp in data['src_kps']:
                    cv2.circle(src_vis, (int(kp[0]), int(kp[1])), 3, (255,0,0), -1)
                    cv2.putText(src_vis, f'({int(kp[0])},{int(kp[1])})', (int(kp[0])+5, int(kp[1])-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                # cv2.imwrite(f'vis_src_{data["src_imname"]}', src_vis)
                
                # Draw target bounding box and keypoints
                trg_vis = trg_img.copy()
                trg_bbox = data['trg_bndbox']
                cv2.rectangle(trg_vis, (trg_bbox[0], trg_bbox[1]), (trg_bbox[2], trg_bbox[3]), (0,0,255), 2)
                # Add bbox coordinates          
                cv2.putText(trg_vis, f'({trg_bbox[0]},{trg_bbox[1]})', (trg_bbox[0], trg_bbox[1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                cv2.putText(trg_vis, f'({trg_bbox[2]},{trg_bbox[3]})', (trg_bbox[2], trg_bbox[3]+15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                # Add keypoint coordinates
                for kp in data['trg_kps']:
                    cv2.circle(trg_vis, (int(kp[0]), int(kp[1])), 3, (255,0,0), -1)
                    cv2.putText(trg_vis, f'({int(kp[0])},{int(kp[1])})', (int(kp[0])+5, int(kp[1])-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                # cv2.imwrite(f'vis_trg_{data["trg_imname"]}', trg_vis)
            
                
            print(f"src_file: {data['src_imname']}, src_img_size: {data['src_imsize']}")
            print(f"trg_file: {data['trg_imname']}, trg_img_size: {data['trg_imsize']}")
            src_img_size = data['src_imsize'][:2][::-1]
            trg_img_size = data['trg_imsize'][:2][::-1]

            src_ft = output_dict[data['src_imname']]
            trg_ft = output_dict[data['trg_imname']]
            
            src_ft = nn.Upsample(size=src_img_size, mode='bilinear')(src_ft)
            trg_ft = nn.Upsample(size=trg_img_size, mode='bilinear')(trg_ft)
            h = trg_ft.shape[-2]
            w = trg_ft.shape[-1]

            trg_bndbox = data['trg_bndbox']
            threshold = max(trg_bndbox[3] - trg_bndbox[1], trg_bndbox[2] - trg_bndbox[0])

            total = 0
            correct = 0

            # Create a combined visualization image
            if VISUALIZE:
                combined_vis = np.hstack((src_vis, trg_vis))

            # for each src_kps find the best match in trg_features
            for idx in range(len(data['src_kps'])): 
                total += 1
                cat_total += 1
                all_total += 1
                src_point = data['src_kps'][idx]
                trg_point = data['trg_kps'][idx]

                num_channel = src_ft.size(1)
                src_vec = src_ft[0, :, src_point[1], src_point[0]].view(1, num_channel) # 1, C
                trg_vec = trg_ft.view(num_channel, -1).transpose(0, 1) # HW, C
                # Apply l2 normalization -> t1=[1,2,3] -> l2norm_t1 = t1/(t1**2).sum()**0.5
                src_vec = F.normalize(src_vec).transpose(0, 1) # c, 1   
                trg_vec = F.normalize(trg_vec) # HW, c
                cos_map = torch.mm(trg_vec, src_vec).view(h, w).cpu().numpy() # H, W

                max_yx = np.unravel_index(cos_map.argmax(), cos_map.shape)
                # cos_map[max_yx] 하면 최대값 구해진다! 

                dist = ((max_yx[1] - trg_point[0]) ** 2 + (max_yx[0] - trg_point[1]) ** 2) ** 0.5
                is_correct = (dist / threshold) <= 0.1
                if is_correct:
                    correct += 1
                    cat_correct += 1
                    all_correct += 1

                # Draw matches on combined visualization
                if VISUALIZE:
                    # Source point coordinates
                    src_x, src_y = int(src_point[0]), int(src_point[1])
                    # Target point coordinates (add source image width for combined image)
                    trg_x = int(max_yx[1]) + src_img.shape[1]  # Add source width offset
                    trg_y = int(max_yx[0])
                    # Draw line between matched points - green for correct, red for incorrect
                    color = (0,255,0) if is_correct else (0,0,255)
                    cv2.line(combined_vis, (src_x, src_y), (trg_x, trg_y), color, 2)  
                    
            if VISUALIZE:
                cv2.imwrite(f'{SAVE_PATH}/src{data["src_imname"]}_trg{data["trg_imname"]}.jpg', combined_vis)
                    
            breakpoint()

            cat_pck.append(correct / total)
        total_pck.extend(cat_pck)

        print(f'{cat} per image PCK@0.1: {np.mean(cat_pck) * 100:.2f}')
        print(f'{cat} per point PCK@0.1: {cat_correct / cat_total * 100:.2f}')
    print(f'All per image PCK@0.1: {np.mean(total_pck) * 100:.2f}')
    print(f'All per point PCK@0.1: {all_correct / all_total * 100:.2f}')
    
    breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPair-71k Evaluation Script')
    parser.add_argument('--dataset_path', type=str, default='./SPair-71k/', help='path to spair dataset')
    parser.add_argument('--save_path', type=str, default='/scratch/lt453/spair_ft/', help='path to save features')
    parser.add_argument('--dift_model', choices=['sd', 'adm'], default='sd', help="which dift version to use")
    parser.add_argument('--img_size', nargs='+', type=int, default=[768, 768],
                        help='''in the order of [width, height], resize input image
                            to [w, h] before fed into diffusion model, if set to 0, will
                            stick to the original input size. by default is 768x768.''')
    parser.add_argument('--t', default=261, type=int, help='t for diffusion')
    parser.add_argument('--up_ft_index', default=1, type=int, help='which upsampling block to extract the ft map')
    parser.add_argument('--ensemble_size', default=8, type=int, help='ensemble size for getting an image ft map')
    args = parser.parse_args()
    main(args)