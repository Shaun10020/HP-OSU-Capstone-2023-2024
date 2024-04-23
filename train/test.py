import logging
from tqdm import tqdm
from torch import nn
import time
import csv
import torch

from utils.convert import convertBinary
from utils.metrics import binary_iou
from ptflops import get_model_complexity_info
from config.config import features, input_height, input_width
'''
This script is reponsible for testing the models
'''
class Test:
    
    def __init__(self,
                 model,
                 device,
                 dataloader,
                 args,
                 criterion = None):
        logging.info("Initializing testing script...")
        '''
        This function is the initialization of testing class
        
        param model: The model architecture, for example, unet. enet, etc.
        param device: The type of device the training is going to take place, torch.device("cuda") or torch.device("cpu")
        param train_dataloader: the training dataloader, contains the training split of the dataset
        param val_dataloader: the validation dataloader, contains the validation split of the dataset
        param args: the argument parser object, contains model information, dataset information, and training hyperparameters
        param optimizer: the optimizer for the model, default is Adam if it is None
        param criterion: the loss function for the model, default is Binary Cross Entropy Loss if it is None 
        '''
        ## Initialization
        self.device = device
        self.args = args
        self.model = model.to(self.device)
        self.test_dataloader = dataloader
        
        ## Setup optimizer and criterion
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.BCELoss()
        logging.info("Done initialize testing script")
        self.pages_per_second = []

        
    def run(self):
        '''
        This function is to run testing on the model with test dataloader
        '''
        logging.info("Running testing script...")
        
        ## Initialization
        self.model.eval()
        epoch_loss = 0.0
        IoU = 0.0
        
        ## Loop through the test dataloader for each batch size
        for batch_data in tqdm(self.test_dataloader):
            inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
            start = time.time()
            preds = self.model(inputs.float())
            preds = torch.sigmoid(preds)
            end = time.time()
            if end - start != 0 :
                self.pages_per_second.append(len(inputs)/(end - start))
            loss = self.criterion(preds,labels)
            epoch_loss += loss.item()
            IoU += binary_iou(convertBinary(preds),labels)
        logging.info(f'''Test Loss: {epoch_loss / len(self.test_dataloader):.4f}''')
        logging.info(f'''Test IoU: {IoU / len(self.test_dataloader)* 100:.2f}%''')
        
        ## Save the results in a csv file
        with open(f'''{self.args.model}-{self.args.dataset}-test.csv''','w') as fd:
            writer = csv.writer(fd)
            writer.writerow(['Test Loss','Test IoU','Test Running Time'])
            writer.writerow([epoch_loss / len(self.test_dataloader),IoU / len(self.test_dataloader)*100,sum(self.pages_per_second) / len(self.pages_per_second)])
        logging.info(f'''Test Executing Time: {sum(self.pages_per_second) / len(self.pages_per_second):.2f} pages per second''')
        logging.info("Done running testing script...")
        
    def flops_count(self):
        macs, params = get_model_complexity_info(self.model, (len(features), input_height, input_width), as_strings=True,
                                           print_per_layer_stat=False, verbose=False)
        return params