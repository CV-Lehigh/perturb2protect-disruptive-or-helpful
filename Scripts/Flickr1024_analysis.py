import os
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
    parser.add_argument('--ori_dataset_path', type=str, default="../Data/Flickr1024_data/ori_gen", help='ori_dataset_path')
    parser.add_argument('--adv_dataset_path', type=str, default="../Data/Flickr1024_data/adv_gen", help='adv_dataset_path')
    parser.add_argument('--save_path', type=str, default="../RealisticImageDomain/style2style/flickr1024_text_img_alignment.pkl", help='save_path')
    return parser.parse_args()

def main(args):

    clip_model, _ = clip.load("ViT-B/32", device=args.device, jit=False)
    clip_model.eval()

    pac_model, pac_preprocess = clip.load("ViT-L/14", device=args.device)
    pac_model = pac_model.float()
    checkpoint = torch.load(args.PACcheckpoint, map_location=args.device)
    pac_model.load_state_dict(checkpoint['state_dict'], strict=False)
    pac_model.eval()
        

    ori_dataset_path = args.ori_dataset_path
    adv_dataset_path = args.adv_dataset_path
    prompts = ["change the style into Cubism",
            "change the style into Post-Impressionism",
            "change the style into Impressionism",
            "change the style into Surrealism",
            "change the style into Baroque",
            "change the style into Fauvism",
            "change the style into Renaissance"]

    results = {}
    for prompt in tqdm(prompts, colour="red", desc="prompts", leave=True):
        results[prompt.rsplit()[-1]] = {}
        input_images, temp_list = [], []
        for image in tqdm(os.listdir(ori_dataset_path), colour="blue", desc="Images", leave=True):
            if image.startswith(prompt.rsplit()[-1]):
                index = image.split("_")[1].split(".")[0]
                temp_list.append(index)
                results[prompt.rsplit()[-1]][index] = {"ori_scores": {}, "adv_scores": {}}
                results[prompt.rsplit()[-1]][index]["ori_scores"] = {}
                results[prompt.rsplit()[-1]][index]["adv_scores"] = {}
                ori_image_path = os.path.join(ori_dataset_path, image)
                adv_image_path = os.path.join(adv_dataset_path, image)
                ori_image = Image.open(ori_image_path)
                adv_image = Image.open(adv_image_path)
                input_images.extend([ori_image, adv_image])
        
        # Compute scores
        clip_score = get_cp_score(clip_model, None, input_images, [prompt] * len(input_images), args.device, "clip")

        ori_clip_scores = clip_score[::2]
        adv_clip_scores = clip_score[1::2]

        pac_score = get_cp_score(pac_model, pac_preprocess, input_images, [prompt] * len(input_images), args.device, "pac")
        ori_pac_scores = pac_score[::2]
        adv_pac_scores = pac_score[1::2]

        # Store results per seed
        for i, indx in enumerate(temp_list):
            # Clip scores
            results[prompt.rsplit()[-1]][indx]["ori_scores"]["clip"] = ori_clip_scores[i]
            results[prompt.rsplit()[-1]][indx]["adv_scores"]["clip"] = adv_clip_scores[i]

            # PAC scores
            results[prompt.rsplit()[-1]][indx]["ori_scores"]["pac"] = ori_pac_scores[i]
            results[prompt.rsplit()[-1]][indx]["adv_scores"]["pac"] = adv_pac_scores[i]

    import pickle
    with open(args.save_path, "wb") as file:
        pickle.dump(results, file)

if __name__ == '__main__':
    args = get_args()
    main(args)