#!/bin/bash

# python main.py --model=unet --mode=train --batch=5 --epoch=70 --dataset=simplex
# python main.py --model=unet --mode=train --batch=5 --epoch=70 --dataset=duplex
# python main.py --model=enet --mode=train --batch=5 --epoch=70 --dataset=simplex
# python main.py --model=enet --mode=train --batch=5 --epoch=120 --dataset=duplex 

# python main.py --model=unet --mode=test --batch=5 --dataset=simplex
# python main.py --model=unet --mode=test --batch=5 --dataset=duplex
# python main.py --model=enet --mode=test --batch=5 --dataset=simplex
# python main.py --model=enet --mode=test --batch=5 --dataset=duplex --parallel=False
python main.py --model=enet --mode=test --batch=5 --dataset=duplex --parallel=True
# python main.py --model=deeplabv3 --mode=test --batch=5 --dataset=duplex
# python main.py --model=deeplabv3+ --mode=test --batch=5 --dataset=duplex