import os
import json
import logging
from tqdm import tqdm
import pandas as pd
from PIL import Image

import torch
from diffusers import StableDiffusionImg2ImgPipeline

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--model_id_or_path', type=str, default="runwayml/stable-diffusion-v1-5", help='model_id_or_path')
    parser.add_argument('--SEED', type=int, default=[9222, 42, 66, 123, 999], help='seeds')
    parser.add_argument('--STRENGTH', type=float, default=0.5, help='STRENGTH')
    parser.add_argument('--GUIDANCE', type=float, default=7.5, help='GUIDANCE')
    parser.add_argument('--NUM_STEPS', type=int, default=50, help='NUM_STEPS')
    parser.add_argument('--caption_path', type=str, default="/home/qit220/Data/metadata.json", help='caption_path')
    parser.add_argument('--ori_dataset_path', type=str, default="/home/qit220/Data/style_transfer_exp/wikiart_subset", help='ori_dataset_path')
    parser.add_argument('--ori_save_path', type=str, default="/home/qit220/Data/style_transfer_exp/wikiart_subset_gen", help='ori_save_path')
    parser.add_argument('--adv_dataset_path', type=str, default="/home/qit220/Data/style_transfer_exp/wikiart_subset_glaze", help='adv_dataset_path')
    parser.add_argument('--adv_save_path', type=str, default="/home/qit220/Data/style_transfer_exp/wikiart_subset_glaze_gen", help='adv_save_path')
    return parser.parse_args()

def main(args):
    
    with open(args.caption_path, "r") as f:
        captions = json.load(f)

    device = args.device

    model_id_or_path = args.model_id_or_path
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id_or_path,
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    pipe_img2img = pipe_img2img.to(device)

    seeds = args.SEED
    STRENGTH = args.STRENGTH
    GUIDANCE = args.GUIDANCE
    NUM_STEPS = args.NUM_STEPS
    ori_dataset_path = args.ori_dataset_path
    ori_save_path = args.ori_save_path
    adv_dataset_path = args.adv_dataset_path
    adv_save_path = args.adv_save_path

    for SEED in tqdm(seeds, colour="green", desc="Seeds", leave=True):
        for caption in tqdm(captions, colour="blue", desc="images", leave=False):
            img_name = caption['Img']
            prompt = "change to style to "+caption['transfer_style']
            ori_image = Image.open(os.path.join(ori_dataset_path, img_name)).resize((512, 512), resample=Image.BICUBIC)
            adv_image = Image.open(os.path.join(adv_dataset_path, img_name)).resize((512, 512), resample=Image.BICUBIC)
            assert ori_image.size == adv_image.size
            with torch.autocast('cuda'):
                torch.manual_seed(SEED)
                image_nat = pipe_img2img(prompt=prompt, image=ori_image, strength=STRENGTH, guidance_scale=GUIDANCE, num_inference_steps=NUM_STEPS).images[0]
                torch.manual_seed(SEED)
                image_adv = pipe_img2img(prompt=prompt, image=adv_image, strength=STRENGTH, guidance_scale=GUIDANCE, num_inference_steps=NUM_STEPS).images[0]
                image_nat.save(os.path.join(ori_save_path, str(SEED) +'_'+ caption['transfer_style']+img_name))
                image_adv.save(os.path.join(adv_save_path, str(SEED) +'_'+ caption['transfer_style']+img_name))

if __name__ == '__main__':
    args = get_args()
    main(args)