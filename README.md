# Disruptive Perturbation
Official code space for "ARE PERTURBATIONS FOR IMAGE PROTECTION DISRUPTIVE TO IMAGE EDITING?"

### Directory Structure

- Data
- Realistic Image Domain
    - img2img
        - Flickr8k original caption
        - Closely-modified caption
        - Extensively-modified caption
    - style2style
        - style transferring prompts
- Artwork Image Domain
    - style transferring prompts
- Scripts
    - SDGen.py (Image Generation by Stable Diffusion)
    - CP_scores.py (functions to call PAC-S++ and CLIP-S metrics which calculate text-image alignment)
    - *_analysis.py (script to compute PAC-S++ and CLIP-S among realistic image domain and artwork image domain, save the output in the format of .pkl)
- environment.yml

### Download and set-up environment
```
git clone https://github.com/CV-Lehigh/PerturbationsDontAlwaysAidProtection.git
cd PerturbationsDontAlwaysAidProtection
conda env create -f environment.yml
```
### Data Preparation 
Download from this [URL](https://drive.google.com/drive/folders/1fxh6ngdv4tYkTqPm2SnnZTabkuIucX8D?usp=drive_link) and put the unzipped folder in /Data.

### Text-Image Alignment Evaluation
```
cd Scripts
python *_analysis.py
```

### Citation
If helpful, please consider citing us as follows:

