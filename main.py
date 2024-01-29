
from config.args import get_arguments
from config.config import train_test_ratio,features,labels
from dataloader.load_data import SimplexDataset
from utils.load_json import load_results
from model.UNet import UNet
from train.train import Train

import logging
import torch

args = get_arguments()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'''Using device: {device}''')
    unet = UNet(len(features),len(labels))
    pdf,algorithm,intermediate = load_results(args.label_folder)
    dataset = SimplexDataset(args.input_folder,args.label_folder,intermediate)
    train = Train(unet,device,dataset,args.batch,args.epoch)
    train.run()