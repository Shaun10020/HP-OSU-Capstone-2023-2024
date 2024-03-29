import logging
from tqdm import tqdm
from torch import nn
import time
import csv

from utils.convert import convertBinary
from utils.metrics import binary_iou
class Test:
    
    def __init__(self,
                 model,
                 device,
                 dataloader,
                 args,
                 criterion = None):
        logging.info("Initializing testing script...")
        self.device = device
        self.args = args
        self.model = model.to(self.device)
        self.test_dataloader = dataloader
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.BCELoss()
        logging.info("Done initialize testing script")
        self.time_per_page = []

        
    def run(self):
        logging.info("Running testing script...")
        self.model.eval()
        epoch_loss = 0.0
        IoU = 0.0
        for batch_data in tqdm(self.test_dataloader):
            inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
            start = time.time()
            preds = self.model(inputs.float())
            end = time.time()
            if end - start != 0 :
                self.time_per_page.append(len(inputs)/(end - start))
            loss = self.criterion(preds,labels)
            epoch_loss += loss.item()
            IoU += binary_iou(convertBinary(preds),labels)
        logging.info(f'''Test Loss: {epoch_loss / len(self.test_dataloader):.4f}''')
        logging.info(f'''Test IoU: {IoU / len(self.test_dataloader)* 100:.2f}%''')
        with open(f'''{self.args.model}-{self.args.dataset}-test.csv''','w') as fd:
            writer = csv.writer(fd)
            writer.writerow(['Test Loss','Test IoU','Test Running Time'])
            writer.writerow([epoch_loss / len(self.test_dataloader),IoU / len(self.test_dataloader)*100,sum(self.time_per_page) / len(self.time_per_page)])
        logging.info(f'''Test Executing Time: {sum(self.time_per_page) / len(self.time_per_page):.2f} pages per second''')
        logging.info("Done running testing script...")