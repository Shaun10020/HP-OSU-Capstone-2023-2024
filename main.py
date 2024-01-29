
from config.args import get_arguments
from config.config import features,labels,duplex_labels,train_test_ratio
from dataloader.load_data import SimplexDataset, DuplexDataset
from utils.load_json import load_results
from model.UNet import UNet
from model.ENet import ENet
from train.train import Train
from train.test import Test

import logging
import torch
from torch.utils.data import random_split

args = get_arguments()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train():
    unet = UNet(2*len(features),2*len(labels)+len(duplex_labels))
    pdf,algorithm,intermediate = load_results(args.label_folder)
    dataset = DuplexDataset(args.input_folder,args.label_folder,intermediate)
    train_set, test_set = random_split(dataset,train_test_ratio,torch.Generator())
    train = Train(unet,device,train_set,args.batch,args.lr)
    train.run(args.epoch)
    
    test = Test(train.model,device,test_set,args.batch)
    test.run()

def test():
    None

def inference():
    None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'''Using device: {device}''')
    if args.mode == 'train':
        train()
    elif args.mode == 'inference':
        inference()
    elif args.mode == 'test':
        test()