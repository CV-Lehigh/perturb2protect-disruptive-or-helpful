import os
import json
from tqdm import tqdm
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


device0 ="cuda:0"
device3 = "cuda:3"


clip_model, _ = clip.load("ViT-B/32", device=device0, jit=False)
clip_model.eval()

pac_model, pac_preprocess = clip.load("ViT-L/14", device=device3)
pac_model = pac_model.float()
checkpoint = torch.load("/home/qit220/pytorch-stable-diffusion/eval/model/PAC++_clip_ViT-L-14.pth", map_location=device3)
pac_model.load_state_dict(checkpoint['state_dict'], strict=False)
pac_model.eval()
    

ori_dataset_path = "/home/qit220/Data/OriginProtectionExp1/check/sd_ori_7p"
adv_dataset_path = "/home/qit220/Data/OriginProtectionExp1/check/sd_adv_wo_target_7p"
prompts = ["change the style into Cubism",
          "change the style into Post-Impressionism",
          "change the style into Impressionism",
          "change the style into Surrealism",
          "change the style into Baroque",
          "change the style into Fauvism",
          "change the style into Renaissance"]

# Initialize a results dictionary
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
    clip_score = get_cp_score(clip_model, None, input_images, [prompt] * len(input_images), device0, "clip")

    ori_clip_scores = clip_score[::2]
    adv_clip_scores = clip_score[1::2]

    pac_score = get_cp_score(pac_model, pac_preprocess, input_images, [prompt] * len(input_images), device3, "pac")
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
with open("pg_1024_score_results.pkl", "wb") as file:
    pickle.dump(results, file)
