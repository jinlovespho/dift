import argparse
import torch
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import os
import json
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

    for cat in all_cats:
        cat_list = cat2json[cat]
        for json_path in tqdm(cat_list):

            with open(os.path.join(dataset_path, test_path, json_path)) as temp_f:
                data = json.load(temp_f)
            
            SAVE_PATH = f'{args.save_vis_pred_kpts_dir}/{cat}'
            if not os.path.exists(SAVE_PATH):
                    os.makedirs(SAVE_PATH)
                    
            VISUALIZE=args.vis_pred_kpts
            ## PHO_VISUALIZE BBOX and KEYPOINTS IN TARGET AND SOURCE
            if VISUALIZE:
                # Load source and target images
                src_img = cv2.imread(os.path.join(dataset_path, 'JPEGImages', cat, data['src_imname']))
                trg_img = cv2.imread(os.path.join(dataset_path, 'JPEGImages', cat, data['trg_imname']))
                
                # Resize target to match source dimensions
                src_h, src_w = src_img.shape[:2]
                trg_h, trg_w = trg_img.shape[:2]
                
                # Calculate scale factors
                scale_x = src_w / trg_w
                scale_y = src_h / trg_h
                
                # Resize target image and adjust coordinates
                trg_img = cv2.resize(trg_img, (src_w, src_h))
                trg_bbox = [int(x * scale_x) if i % 2 == 0 else int(x * scale_y) 
                           for i, x in enumerate(data['trg_bndbox'])]
                trg_kps = [[int(kp[0] * scale_x), int(kp[1] * scale_y)] 
                          for kp in data['trg_kps']]
                
                # Draw source bounding box and keypoints
                src_vis = src_img.copy()
                src_bbox = data['src_bndbox']
                cv2.rectangle(src_vis, (src_bbox[0], src_bbox[1]), (src_bbox[2], src_bbox[3]), (255,0,0), 2)
                
                # Draw target bounding box and keypoints
                trg_vis = trg_img.copy()
                cv2.rectangle(trg_vis, (trg_bbox[0], trg_bbox[1]), (trg_bbox[2], trg_bbox[3]), (255,0,0), 2)
                
                # Create a combined visualization
                combined_vis = np.hstack((src_vis, trg_vis))
                
                # Draw ground truth matching keypoints with lines connecting them
                for idx, (src_kp, trg_kp) in enumerate(zip(data['src_kps'], trg_kps)):
                    # Draw source keypoint
                    src_pt = (int(src_kp[0]), int(src_kp[1]))
                    cv2.circle(combined_vis, src_pt, 5, (0,0,255), -1)  # Green dots for keypoints

                    # Draw target keypoint
                    trg_pt = (int(trg_kp[0] + src_w), int(trg_kp[1]))  # Add src_w offset for target points
                    cv2.circle(combined_vis, trg_pt, 5, (0,0,255), -1)

                    # Draw line connecting ground truth matches
                    cv2.line(combined_vis, src_pt, trg_pt, (0,255,0), 1)  # White lines for connections
                
                # Save visualization
                cv2.imwrite(f"{SAVE_PATH}/GT_src{data['src_imname'].split('.')[0]}_trg{data['trg_imname'].split('.')[0]}.jpg", combined_vis)


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
    
    parser.add_argument('--vis_pred_kpts', action='store_true')
    parser.add_argument('--save_vis_pred_kpts_dir', type=str)
    args = parser.parse_args()
    main(args)