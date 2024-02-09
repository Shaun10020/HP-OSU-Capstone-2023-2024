import logging
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
import os

from utils.convert import convertBinary
from utils.metrics import binary_iou
from config.config import pin_memory
class Test:
    
    def __init__(self,
                 model,
                 device,
                 dataset,
                 batch_size,
                 criterion = None):
        logging.info("Initializing testing script...")
        self.device = device
        self.model = model.to(self.device)
        loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=pin_memory)
        self.test_dataloader = DataLoader(dataset, shuffle=True, **loader_args)
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()
        logging.info("Done initialize testing script")

        
    def run(self):
        logging.info("Running testing script...")
        self.model.eval()
        epoch_loss = 0.0
        IoU = 0.0
        for batch_data in tqdm(self.test_dataloader):
            inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
            preds = self.model(inputs.float())
            loss = self.criterion(labels,preds)
            epoch_loss += loss.item()
            IoU += binary_iou(convertBinary(preds),labels)
        logging.info(f'''Test Loss: {epoch_loss / len(self.test_dataloader):.4f}''')
        logging.info(f'''Test IoU: {IoU / len(self.test_dataloader * 100):.2f}%''')
        logging.info("Done running testing script...")