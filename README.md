# fsr_unet
discribe what to achieve for this repo  
treat the code as a communication tool with others, which means:  
- others should understand the code  
- others should be able to run the code, and env is very important  
Good Job, Zhangying

# Abstract
This repo is a baseline experiment for isotropic super-resolution project based on  "Deep Learning for Isotropic Super-Resolution from Non-isotropic 3D Electron Microscopy".  The experiment is trained on GPU(NVIDIA 2080Ti).
## dataset
use "cremi" and cut a large 3D image into several 100×128×128 small pieces.
## install
install essential packages
```
pip install -r requirements.txt
```
## files 
"model_fsrcnn.py" is the network structure of 3D-FSRCNN  
"model_3dunet.py" is the network structure of 3D-SR-UNet based on 3D-UNet  
"dataloader.py"  is the data loading and preprocessing process  
"train_2.py" is the training and testing implementation process  
## run
run "train_2.py"
