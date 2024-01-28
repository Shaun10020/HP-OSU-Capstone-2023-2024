import logging
from tqdm import tqdm
from torch import optim
from torch import nn

from config.config import train_val_ratio,lr
from dataloader.load_data import load_dataloader

class Train:
    
    def __init__(self,
                 model,
                 device,
                 dataset,
                 batch_size,
                 optimizer = None,
                 criterion = None):
        self.model = model
        self.device = device
        self.train_dataloader, self.val_dataloader = load_dataloader(dataset,batch_size,train_val_ratio)
        if optimizer:
            self.optim = optimizer
        else:
            self.optim = optim.Adam(model.parameters(), lr=lr)
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()

        
    def run_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        for step, batch_data in enumerate(self.train_dataloader):
            None
        
    def validate(self):
        None