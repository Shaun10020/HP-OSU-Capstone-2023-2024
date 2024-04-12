# HP-OSU-Capstone-2023-2024

The purpose of this repository is to train a semantic segmentation model to predict characteristic binary mask images. This repository uses pytorch framework. The models being used are UNet, ENet and DeepLabV3+

- Requirements
- Quick Start
	- how to train 
	- how to test
	- how to inference
- Help
- Project structure
---
## Requirements
```
matplotlib >= 3.8.2
Python >= 3.11.7
numpy >= 1.26.3
pillow >= 10.2.0
torch >= 2.2.0+cu118
torchaudio >= 2.2.0+cu118
torchvision >= 0.17.0+cu118
tqdm >= 4.66.1
```
---
## Quick start

#### How to train
Example:
`python main.py --model=unet --mode=train --batch=5 --epoch=70 --dataset=simplex`
#### How to test
Example:
`python main.py --model=unet --mode=test --batch=5 --dataset=simplex`
#### How to inference
Example:
`python main.py --model=unet --mode=inference --batch=5 --dataset=simplex`

---
## Help
This commands list all available arguments for running main.py
`python main.py --help`

---
## Project structure

- checkpoints
- config
- data
- dataloader
- model
- pdf_webcrawler
- plots
- train
- utils

**checkpoints** :This folder contains the files that store model weights
**config**: This folder contains the configuration related to input channels and output characteristics
**data**: This is where the data should be place under
**dataloader**: This folder contains a script to process and convert data to dataloader , and save them as .pt files
**model**: This folder contains the model architectures
**pdf_webcrawler**: This folder contains scripts that use scrappy to scrap pdf files from websites
**plots**: This folder contains plots, such as training loss over epoch.
**train**: This folder contains training script, testing script
**utils**: This folder contains utilities scripts, such as load json files, check for valid data, etc.