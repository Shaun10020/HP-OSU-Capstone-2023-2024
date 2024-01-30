
from config.args import get_arguments
from config.config import features,labels,duplex_labels,train_test_ratio
from dataloader.load_data import SimplexDataset, DuplexDataset
from utils.load_json import load_results
from utils.save_load_model import load
from model.UNet import UNet
from model.ENet import ENet
from model.DeepLabV3 import CustomDeepLabV3
from train.train import Train
from train.test import Test

import logging
import torch
from torch.utils.data import random_split

args = get_arguments()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(model,dataset):
    train_set, test_set = random_split(dataset,train_test_ratio,torch.Generator())
    train = Train(model,device,train_set,args)
    train.run(args.epoch)
    
    model = load(model,args)
    test = Test(model,device,test_set,args.batch)
    test.run()

def test(model,dataset):
    pdf,algorithm,intermediate = load_results(args.label_folder)
    model = load(model,args)
    test = Test(model,device,dataset,args.batch)
    test.run()

def inference(model):
    None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'''Using device: {device}''')
    
    n_input = len(features) if args.dataset == 'simplex' else 2*len(features)
    n_output = len(labels) if args.dataset == 'simplex' else 2*len(labels)+len(duplex_labels)
    if args.mode != 'inference':
        pdf,algorithm,intermediate = load_results(args.label_folder)
        if args.dataset =='simplex':
            dataset = SimplexDataset(args.input_folder,args.label_folder,intermediate)
        else:
            dataset = DuplexDataset(args.input_folder,args.label_folder,intermediate)
            
    if args.model == 'unet':
        model = UNet(n_input,n_output)
    elif args.model == 'enet':
        model = ENet(n_input,n_output)
    elif args.model == 'deeplabv3':
        model = CustomDeepLabV3(n_output)
    
    if args.mode == 'train':
        train(model,dataset)
    elif args.mode == 'inference':
        inference(model)
    elif args.mode == 'test':
        test(model,dataset)
