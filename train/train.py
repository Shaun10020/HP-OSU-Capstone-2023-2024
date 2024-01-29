import logging
from tqdm import tqdm
import torch
import os

from config.config import train_val_ratio
from dataloader.load_data import load_dataloader
from utils.save_load_model import save

class Train:
    
    def __init__(self,
                 model,
                 device,
                 dataset,
                 args,
                 optimizer = None,
                 criterion = None):
        logging.info("Initializing training script...")
        self.device = device
        self.args = args
        self.model = model.to(self.device)
        self.train_dataloader, self.val_dataloader = load_dataloader(dataset,args.batch,train_val_ratio)
        if optimizer:
            self.optim = optimizer
        else:
            self.optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        self.epoch_losses = []
        self.epoch_losses_val = []
        logging.info("Done initialize training script")

        
    def run_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        for batch_data in tqdm(self.train_dataloader):
            self.optim.zero_grad()
            inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
            preds = self.model(inputs.float())
            loss = self.criterion(labels,preds)
            loss.backward()
            self.optim.step()
            epoch_loss += loss.item()
        self.epoch_losses.append(epoch_loss / len(self.train_dataloader))
        
    def run_epoch_val(self):
        self.model.eval()
        epoch_loss = 0.0
        for batch_data in tqdm(self.val_dataloader):
            inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
            preds = self.model(inputs.float())
            loss = self.criterion(labels,preds)
            epoch_loss += loss.item()
        self.epoch_losses_val.append(epoch_loss / len(self.val_dataloader))
            
    def run(self,epochs):
        logging.info("Running training script...")
        for i in range(epochs):
            logging.info(f'''Running Epoch {i+1}/{epochs}...''')
            self.run_epoch()
            self.run_epoch_val()
            logging.info(f'''Epoch [{i+1}/{epochs}], Loss: {self.epoch_losses[-1]:.4f}, Val Loss: {self.epoch_losses_val[-1]:.4f}''')
            if self.epoch_losses_val[-1] == min(self.epoch_losses_val):
                save(self.model,self.args)
        logging.info("Done running training script...")