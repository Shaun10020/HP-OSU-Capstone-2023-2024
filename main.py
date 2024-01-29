
from config.args import get_arguments
from config.config import features,labels,duplex_labels,train_test_ratio
from dataloader.load_data import SimplexDataset, DuplexDataset
from utils.load_json import load_results
from utils.save_load_model import load
from model.UNet import UNet
from model.ENet import ENet
from train.train import Train
from train.test import Test

import logging
import torch
from torch.utils.data import random_split
<<<<<<< HEAD
import os
=======
>>>>>>> 606262a98b8e14486d4def8ba36ae016bff1eae3

args = get_arguments()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(n_input,n_output,dataset):
    model = UNet(n_input,n_output)
    train_set, test_set = random_split(dataset,train_test_ratio,torch.Generator())
    train = Train(model,device,train_set,args)
    train.run(args.epoch)
    
    model = load(model,args)
    test = Test(model,device,test_set,args.batch)
    test.run()

def test(n_input,n_output,dataset):
    model = UNet(n_input,n_output)
    pdf,algorithm,intermediate = load_results(args.label_folder)
    model = load(model,args)
    test = Test(model,device,dataset,args.batch)
    test.run()

def inference():
    None

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
<<<<<<< HEAD
    
    n_input = len(features) if args.dataset == 'simplex' else 2*len(features)
    n_output = len(labels) if args.dataset == 'simplex' else 2*len(labels)+len(duplex_labels)
    if args.mode != 'inference':
        pdf,algorithm,intermediate = load_results(args.label_folder)
        if args.dataset =='simplex':
            dataset = SimplexDataset(args.input_folder,args.label_folder,intermediate)
        else:
            dataset = DuplexDataset(args.input_folder,args.label_folder,intermediate)
            
    if args.mode == 'train':
        train(n_input,n_output,dataset)
    elif args.mode == 'inference':
        inference()
    elif args.mode == 'test':
        test(n_input,n_output,dataset)
=======
    if args.mode == 'train':
        train()
    elif args.mode == 'inference':
        inference()
    elif args.mode == 'test':
        test()
>>>>>>> 606262a98b8e14486d4def8ba36ae016bff1eae3
