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
                 epochs,
                 optimizer = None,
                 criterion = None):
        logging.info("Initializing training script...")
        self.device = device
        self.model = model.to(self.device)
        self.epochs = epochs
        self.train_dataloader, self.val_dataloader = load_dataloader(dataset,batch_size,train_val_ratio)
        if optimizer:
            self.optim = optimizer
        else:
            self.optim = optim.Adam(model.parameters(), lr=lr)
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.epoch_losses = []
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
        self.epoch_losses.append(epoch_loss / len(self.train_loader))
            
    def run(self):
        logging.info("Running training script...")
        self.model.train()
        for i in range(self.epochs):
            logging.info(f'''Running Epoch {i}/{self.epochs}...''')
            self.run_epoch()
        logging.info("Done running training script...")