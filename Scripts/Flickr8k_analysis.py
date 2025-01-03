import os
import json
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import clip
import torch

from CP_scores import get_cp_score

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Stable Diffusion generation.')
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--PACcheckpoint', type=str, default="/PAC++_clip_ViT-L-14.pth", help='model_id_or_path')
    parser.add_argument('--SEED', type=int, default=[9222, 42, 66, 123, 999], help='seeds')
    parser.add_argument('--caption_path', type=str, default="../RealisticImageDomain/Closely_modified_caption.json", help='caption_path')
    parser.add_argument('--ori_dataset_path', type=str, default="../Data/Flickr8k_data/Flickr8k_ori_gen", help='ori_dataset_path')
    parser.add_argument('--adv_dataset_path', type=str, default="../Data/Flickr8k_data/Flickr8k_adv_gen", help='adv_dataset_path')
    parser.add_argument('--save_path', type=str, default="../RealisticImageDomain/img2img/close_caption_text_img_alignment.pkl", help='save_path')
    return parser.parse_args()

def main(args):
    with open(args.caption_path, "r") as f:
        captions = json.load(f)

    clip_model, _ = clip.load("ViT-B/32", device=args.device, jit=False)
    clip_model.eval()

    pac_model, pac_preprocess = clip.load("ViT-L/14", device=args.device)
    pac_model = pac_model.float()
    checkpoint = torch.load(args.PACcheckpoint, map_location=args.device)
    pac_model.load_state_dict(checkpoint['state_dict'], strict=False)
    pac_model.eval()

    results = {}
    for _, (img, caption) in tqdm(enumerate(captions.items()), colour="blue", desc="images", leave=False):
        img_name = img
        prompt = caption
        results[img_name] = {}
        input_images = []
        for seed in tqdm(args.SEED, colour="green", desc="Seeds", leave=True):
            ori_image = Image.open(os.path.join(args.ori_dataset_path, str(seed) +'_'+ img_name + '.png'))
            adv_image = Image.open(os.path.join(args.adv_dataset_path, str(seed) +'_'+ img_name + '.png'))

            results[img_name] = {"ori_scores": {}, "adv_scores": {}}
            input_images.extend([ori_image, adv_image])
        
        # Compute scores
        clip_score = get_cp_score(clip_model, None, input_images, [prompt] * len(input_images), args.device, "clip")

        ori_clip_scores = clip_score[::2]
        adv_clip_scores = clip_score[1::2]

        pac_score = get_cp_score(pac_model, pac_preprocess, input_images, [prompt] * len(input_images), args.device, "pac")
        ori_pac_scores = pac_score[::2]
        adv_pac_scores = pac_score[1::2]

        for i, seed in enumerate(args.SEED):
            # Clip scores
            results[img_name]["ori_scores"]["Seed_"+seed]["clip"] = ori_clip_scores[i]
            results[img_name]["adv_scores"]["Seed_"+seed]["clip"] = adv_clip_scores[i]

            # PAC scores
            results[img_name]["ori_scores"]["Seed_"+seed]["pac"] = ori_pac_scores[i]
            results[img_name]["adv_scores"]["Seed_"+seed]["pac"] = adv_pac_scores[i]

    import pickle
    with open(args.save_path, "wb") as file:
        pickle.dump(results, file)

if __name__ == '__main__':
    args = get_args()
    main(args)