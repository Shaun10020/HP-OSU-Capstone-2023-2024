#!/bin/bash

python main.py --model=unet --mode=train --dataset=simplex --epoch=150 --batch=5
python main.py --model=unet --mode=train --dataset=duplex --epoch=150 --batch=5