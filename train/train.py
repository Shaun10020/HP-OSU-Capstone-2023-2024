import logging

from dataloader.load_data import load_dataloader

def train_model(model,device,dataset,epoch,batch_size):
    # 1. Prepare Dataset and Dataloader (actually, no, just prepare dataloader)
    # 2. Build model and load weights if available (No, should be handled by main script)
    # 3. train the model 
    # 4. save the model
    train_dataloader, val_dataloader = load_dataloader(dataset,batch_size)