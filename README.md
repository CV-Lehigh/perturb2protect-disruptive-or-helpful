Official code space for paper "Is Perturbation-based Image Protection Disruptive to Image Editing?"

### Abstract
The remarkable image generation capabilities of state-of-the-art diffusion models, such as Stable Diffusion,can also be misused to spread misinformation and plagiarize copyrighted materials. To mitigate the potential risks associated with image editing, current image protection methods rely on adding imperceptible perturbations to images to obstruct diffusion-based editing. A fully successful protection for an image implies that the output of editing attempts is an undesirable, noisy image which is completely unrelated to the reference image. In our experiments with various perturbation-based image protection methods across multiple domains (natural scene images and artworks) and editing tasks (image-to-image generation and style editing), we discover that such protection does not achieve this goal completely. In most scenarios, diffusion-based editing of protected images generates a desirable output image which adheres precisely to the guidance prompt. Our findings suggest that adding noise to images may paradoxically increase their association with given text prompts during the generation process, leading to unintended consequences such as better resultant edits. Hence, we argue that perturbation-based methods may not provide a sufficient solution for robust image protection against diffusion-based editing. 

### Directory Structure

- Data
- Natural Scene Image Domain
    - img2img
        - Flickr8k original caption
        - Closely-modified caption
        - Extensively-modified caption
    - stylization
        - style transferring prompts
- Artwork Image Domain
    - style transferring prompts
- Scripts
    - SDGen.py (Image Generation by Stable Diffusion)
    - CP_scores.py (functions to call PAC-S++ and CLIP-S metrics which calculate text-image alignment)
    - *_analysis.py (script to compute PAC-S++ and CLIP-S among realistic image domain and artwork image domain, save the output in the format of .pkl)
- environment.yml

### Set-up environment
```
conda env create -f environment.yml
```
### Data Preparation 
Download from this [URL](https://drive.google.com/drive/folders/1fxh6ngdv4tYkTqPm2SnnZTabkuIucX8D?usp=drive_link) and put the unzipped folder in /Data.

### Model Preparation

Download PAC-S++ model checkpoint from [PAC++_clip_ViT-L-14.pth](https://ailb-web.ing.unimore.it/publicfiles/pac++/PAC++_clip_ViT-L-14.pth) and put it in /model folder.

### Text-Image Alignment Evaluation
```
cd Scripts
python *_analysis.py
```

### Citation
If helpful, please consider citing us as follows:
```
@article{tang2025perturbation,
  title={Is Perturbation-Based Image Protection Disruptive to Image Editing?},
  author={Tang, Qiuyu and Ayambem, Bonor and Chuah, Mooi Choo and Bharati, Aparna},
  journal={2025 IEEE International Conference on Image Processing (ICIP)},
  year={2025}
}
```