import logging
from tqdm import tqdm

from dataloader.load_data import load_dataloader

class Test:
    
    def __init__(self,model,device,dataloader,batch_size):
        self.model = model
        self.device = device
        self.test_dataloader = dataloader
        
    def run_epoch(self):
        None
        
    def validate(self):
        None