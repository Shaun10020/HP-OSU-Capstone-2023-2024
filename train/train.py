import logging
from tqdm import tqdm

from config.config import train_val_ratio
from dataloader.load_data import load_dataloader

class Train:
    
    def __init__(self,model,device,dataset,batch_size):
        self.model = model
        self.device = device
        self.train_dataloader, self.val_dataloader = load_dataloader(dataset,batch_size,train_val_ratio)
        
    def run_epoch(self):
        None
        
    def validate(self):
        None