
from config.args import get_arguments
from config.config import train_test_ratio
from dataloader.load_data import SimplexDataset,DuplexDataset, load_dataloader
from utils.load_json import load_results
from model.UNet import UNet
from train.train import Train

import torch

args = get_arguments()
device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cuda")


if __name__ == "__main__":
    unet = UNet
    pdf,algorithm,intermediate = load_results(args.label_folder)
    dataset = SimplexDataset(args.input_folder,args.label_folder,intermediate)
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    # dataset2 = DuplexDataset(args.input_folder,args.label_folder,intermediate)
    # train_dataloader , test_dataloader = load_dataloader(dataset,args.batch,train_test_ratio)
    # train_dataloader2 , test_dataloader2 = load_dataloader(dataset2,args.batch,train_test_ratio)
    # for batch in test_dataloader:
    #     print(len(batch[input_label]))
    # for batch in test_dataloader2:
    #     print(len(batch[input_label]))
    # print("done")
    # train = Train(unet,device,dataset,args.batch)