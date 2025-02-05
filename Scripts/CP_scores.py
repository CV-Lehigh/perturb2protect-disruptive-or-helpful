'''
Code for CLIPScore (https://arxiv.org/abs/2104.08718)
@inproceedings{hessel2021clipscore,
  title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
  author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
  booktitle={EMNLP},
  year={2021}
}
'''
'''
Code from PAC-Score (https://github.com/aimagelab/pacscore)
@inproceedings{sarto2023positive,
  title={{Positive-Augmented Contrastive Learning for Image and Video Captioning Evaluation}},
  author={Sarto, Sara and Barraco, Manuele and Cornia, Marcella and Baraldi, Lorenzo and Cucchiara, Rita},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
@inproceedings{sarto2024positive,
  title={{Positive-Augmented Contrastive Learning for Vision-and-Language Evaluation and Training}},
  author={Sarto, Sara and Nicholas, Moratelli and Cornia, Marcella and Baraldi, Lorenzo and Cucchiara, Rita},
  booktitle={arxiv},
  year={2024}
}
'''

import clip
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import tqdm
import numpy as np
import sklearn.preprocessing
import collections
import warnings
from packaging import version



class CapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix='A photo depicts'):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        if transform:
            self.preprocess = transform
        else:
            # CLIP-S: only 224x224 ViT-B/32 supported for now
            self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        image = self.data[idx]
        image = self.preprocess(image)
        return {'image':image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=800, num_workers=8):
    data = torch.utils.data.DataLoader(
        CapDataset(captions),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, transform, device, batch_size=800, num_workers=8):
    data = torch.utils.data.DataLoader(
        ImageDataset(images, transform),
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False
        )
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['image'].to(device)
            # if device == 'cuda':
            #     b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def get_cp_score(model, transform, images, candidates, device, flag):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(images, list):
        # need to extract image features
        images = extract_all_images(images, model, transform, device)
    
    candidates = extract_all_captions(candidates, model, device)

    #as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        # warnings.warn(
        #     'due to a numerical instability, new numpy normalization is slightly different than paper results. '
        #     'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))
    if flag == 'pac':
        w = 2.0
    elif flag == 'clip':
        w = 2.5
    score = w*np.clip(np.sum(images * candidates, axis=1), 0, None)
    return score



def pac_clip_score(images: list[Image], prompts: list[str], device, flag) -> list[float]:


    if flag == 'clip':
        model, _ = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()
        preprocess = None
    elif flag == 'pac':
        model, preprocess = clip.load("ViT-L/14", device=device)
        model = model.to(device)
        model = model.float()
        checkpoint = torch.load("./model/PAC++_clip_ViT-L-14.pth", map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.eval()
    
    # get image-text clipscore
    per_instance_image_text = get_cp_score(
        model, preprocess, images, prompts, device, flag)

    scores = [np.mean(float(clipscore))
                for clipscore in per_instance_image_text]
    
    return scores
