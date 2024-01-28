import logging
from tqdm import tqdm
from torch import optim
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler

from config.config import train_val_ratio, weight_decay,lr,lr_decay,lr_decay_epochs
from dataloader.load_data import load_dataloader

class Train:
    
    def __init__(self,
                 model,
                 device,
                 dataset,
                 batch_size,
                 optimizer = None,
                 lr_updater = None,
                 criterion = None):
        self.model = model
        self.device = device
        self.train_dataloader, self.val_dataloader = load_dataloader(dataset,batch_size,train_val_ratio)
        if optimizer:
            self.optim = optimizer
        else:
            self.optim = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
        if lr_updater:
            self.lr_updater = lr_updater  
        else:
            self.lr_updater = lr_scheduler.StepLR(self.optim, lr_decay_epochs,lr_decay)
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()

        
    def run_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        for step, batch_data in enumerate(self.data_loader):
            None
        
    def validate(self):
        None