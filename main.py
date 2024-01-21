
from config.args import get_arguments
from config.config import input_label
from dataloader.load_data import SimplexDataset,load_dataloader
from utils.load_json import load_results
from utils.check_data import checkResult

import torch

args = get_arguments()
device = torch.device("cpu") if torch.cuda.is_available() else  torch.device("cuda")


if __name__ == "__main__":
    pdf,algorithm,intermediate = load_results(args.label_folder)
    intermediate = checkResult(args.input_folder,args.label_folder,intermediate)
    dataset = SimplexDataset(args.input_folder,args.label_folder,intermediate)
    print(len(dataset))
    train_dataloader,val_dataloader = load_dataloader(dataset)