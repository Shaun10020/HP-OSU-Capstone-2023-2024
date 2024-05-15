#!/bin/bash

## Train the models
python main.py --model=unet --mode=train --batch=8 --epoch=120 --dataset=duplex --lr=1e-4 --lr_decay=0.5 --lr_epoch=30
python main.py --model=enet --mode=train --batch=8 --epoch=120 --dataset=duplex --lr=1e-4 --lr_decay=0.5 --lr_epoch=30
python main.py --model=deeplabv3 --mode=train --batch=8 --epoch=120 --dataset=duplex --lr=1e-4 --lr_decay=0.5 --lr_epoch=30
python main.py --model=deeplabv3+ --mode=train --batch=8 --epoch=120 --dataset=duplex --lr=1e-4 --lr_decay=0.5 --lr_epoch=30

## Test models with GPU
python main.py --model=unet --mode=test --batch=8 --dataset=duplex
python main.py --model=enet --mode=test --batch=8 --dataset=duplex
python main.py --model=deeplabv3 --mode=test --batch=8 --dataset=duplex
python main.py --model=deeplabv3+ --mode=test --batch=8 --dataset=duplex

## Test models with CPU
python main.py --model=unet --mode=test --batch=8 --dataset=duplex --device=cpu
python main.py --model=enet --mode=test --batch=8 --dataset=duplex --device=cpu
python main.py --model=deeplabv3 --mode=test --batch=8 --dataset=duplex --device=cpu
python main.py --model=deeplabv3+ --mode=test --batch=8 --dataset=duplex --device=cpu

## Test models with CPU and parallel processes
python main.py --model=unet --mode=test --batch=8 --dataset=duplex --device=cpu --parallel=True
python main.py --model=enet --mode=test --batch=8 --dataset=duplex --device=cpu --parallel=True
python main.py --model=deeplabv3 --mode=test --batch=8 --dataset=duplex --device=cpu --parallel=True
python main.py --model=deeplabv3+ --mode=test --batch=8 --dataset=duplex --device=cpu --parallel=True
