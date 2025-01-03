import os
import json
import logging
from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import clip
import torch

from CP_scores import get_cp_score

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float16):
            return float(obj)
        # Add handling for other types if necessary
        return super().default(obj)


device0 ="cuda:2"
device3 = "cuda:3"

with open("/home/qit220/Data/metadata.json", "r") as f:
    captions = json.load(f)

clip_model, _ = clip.load("ViT-B/32", device=device0, jit=False)
clip_model.eval()

pac_model, pac_preprocess = clip.load("ViT-L/14", device=device3)
pac_model = pac_model.float()
checkpoint = torch.load("/home/qit220/pytorch-stable-diffusion/eval/model/PAC++_clip_ViT-L-14.pth", map_location=device3)
pac_model.load_state_dict(checkpoint['state_dict'], strict=False)
pac_model.eval()
    
seeds = [9222, 42, 66, 123, 999]

ori_dataset_path = "/home/qit220/Data/style_transfer_exp/wikiart_subset_gen"
m_adv_dataset_path = "/home/qit220/Data/style_transfer_exp/wikiart_subset_mist_gen"
g_adv_dataset_path = "/home/qit220/Data/style_transfer_exp/wikiart_subset_glaze_gen"
results = {}
for caption in tqdm(captions, colour="blue", desc="images", leave=False):
    img_name = caption['Img']
    prompt = "change to style to "+caption['transfer_style']
    results[img_name] = {}
    input_images = []
    for seed in tqdm(seeds, colour="green", desc="Seeds", leave=True):
        ori_image = Image.open(os.path.join(ori_dataset_path, str(seed) +'_'+ caption['transfer_style']+img_name))
        m_adv_image = Image.open(os.path.join(m_adv_dataset_path, str(seed) +'_'+ caption['transfer_style']+img_name))
        g_adv_image = Image.open(os.path.join(g_adv_dataset_path, str(seed) +'_'+ caption['transfer_style']+img_name))
        assert ori_image.size == m_adv_image.size == g_adv_image.size
        results[img_name][seed] = {"ori_scores": {}, "m_adv_scores": {}, "g_adv_scores": {}}
        input_images.extend([ori_image, m_adv_image, g_adv_image])
    
    # Compute scores
    clip_score = get_cp_score(clip_model, None, input_images, [prompt] * len(input_images), device0, "clip")

    ori_clip_scores = clip_score[::3]
    m_adv_clip_scores = clip_score[1::3]
    g_adv_clip_scores = clip_score[2::3]

    pac_score = get_cp_score(pac_model, pac_preprocess, input_images, [prompt] * len(input_images), device3, "pac")
    ori_pac_scores = pac_score[::3]
    m_adv_pac_scores = pac_score[1::3]
    g_adv_pac_scores = pac_score[2::3]

    for i, seed in enumerate(seeds):
        # Clip scores
        results[img_name][seed]["ori_scores"]["clip"] = ori_clip_scores[i]
        results[img_name][seed]["m_adv_scores"]["clip"] = m_adv_clip_scores[i]
        results[img_name][seed]["g_adv_scores"]["clip"] = g_adv_clip_scores[i]

        # PAC scores
        results[img_name][seed]["ori_scores"]["pac"] = ori_pac_scores[i]
        results[img_name][seed]["m_adv_scores"]["pac"] = m_adv_pac_scores[i]
        results[img_name][seed]["g_adv_scores"]["pac"] = g_adv_pac_scores[i]


import pickle
with open("mist_glaze_wikiart_score_results.pkl", "wb") as file:
    pickle.dump(results, file)
