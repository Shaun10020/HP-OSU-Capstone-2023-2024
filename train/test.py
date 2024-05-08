import logging
from tqdm import tqdm
from torch import nn
import time
import csv
import torch
import os

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
                 rank = 0,
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
        param rank: The rank number, default is 0.
        param criterion: the loss function for the model, default is Binary Cross Entropy Loss if it is None 
        '''
        ## Initialization
        self.device = device
        self.args = args
        self.model = model.to(self.device)
        self.rank = rank
        self.test_dataloader = dataloader
        
        ## Setup optimizer and criterion
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.BCELoss()
        logging.info("Done initialize testing script")
        self.pages_per_second = []
        self.times = []
        self.epoch_loss = 0
        self.IoU = 0.0

    def run(self):
        if self.device == torch.device("cuda"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.run_cuda()
            end.record()
            torch.cuda.synchronize()
            _time = start.elapsed_time(end) / 1000
        else:
            start = time.time()
            self.run_CPU()
            end = time.time()
            _time = end - start
        macs, params = self.flops_count()
        logging.info(f'''Test Loss: {self.epoch_loss / len(self.test_dataloader):.4f}''')
        logging.info(f'''Test IoU: {self.IoU / len(self.test_dataloader)* 100:.2f}%''')
        logging.info(f'''Number of parameters: {params}''')
        logging.info(f'''Computational complexity: {macs}''')
        
        ## Save the results in a csv file
        if not os.path.exists(f'''{self.args.model}-{self.args.dataset}-test-{"CUDA" if self.device==torch.device("cuda") else "CPU"}.csv'''):
            with open(f'''{self.args.model}-{self.args.dataset}-test-{"CUDA" if self.device==torch.device("cuda") else "CPU"}.csv''','w',newline='') as fd:
                writer = csv.writer(fd)
                writer.writerow(['Rank','Test Loss','Test IoU','Test Inference Time','Test Inference Speed','Test Overall'])
        with open(f'''{self.args.model}-{self.args.dataset}-test-{"CUDA" if self.device==torch.device("cuda") else "CPU"}.csv''','a',newline='') as fd:
            writer = csv.writer(fd)
            writer.writerow([self.rank,self.epoch_loss / len(self.test_dataloader),self.IoU / len(self.test_dataloader)*100,sum(self.times) / len(self.times),sum(self.pages_per_second) / len(self.pages_per_second),_time])
        logging.info(f'''Test Inference Time: {sum(self.times) / len(self.times):.2f} second''')
        logging.info(f'''Test Inference Speed: {sum(self.pages_per_second) / len(self.pages_per_second):.2f} pages per second''')
        logging.info("Done running testing script...")
        
    def run_CPU(self):
        '''
        This function is to run testing on the model with test dataloader
        '''
        logging.info("Running testing script...")
        
        ## Initialization
        self.model.eval()
        self.epoch_loss = 0.0
        self.IoU = 0.0
        text = f'#{self.rank}'
        
        ## Loop through the test dataloader for each batch size
        for batch_data in tqdm(self.test_dataloader,desc=text,position=self.rank):
            inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
            start = time.time()
            preds = self.model(inputs.float())
            preds = torch.sigmoid(preds)
            end = time.time()
            _time = end - start
            if _time != 0 :
                self.pages_per_second.append(len(inputs)/(_time))
                self.times.append(_time)
            loss = self.criterion(preds,labels)
            self.epoch_loss += loss.item()
            self.IoU += binary_iou(convertBinary(preds),labels)
        

        
    def run_cuda(self):
        '''
        This function is to run testing on the model with test dataloader
        '''
        logging.info("Running testing script...")
        
        ## Initialization
        self.model.eval()
        self.epoch_loss = 0.0
        self.IoU = 0.0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        text = f'#{self.rank}'
        
        ## GPU warmup
        batch_data_0, batch_data_1 = next(iter(self.test_dataloader))
        inputs, labels = batch_data_0.to(self.device), batch_data_1.to(self.device)
        preds = self.model(inputs.float())
        
        ## Loop through the test dataloader for each batch size
        for batch_data in tqdm(self.test_dataloader,desc=text,position=self.rank):
            inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
            start.record()
            preds = self.model(inputs.float())
            preds = torch.sigmoid(preds)
            end.record()
            torch.cuda.synchronize()
            time = start.elapsed_time(end) / 1000
            if time != 0 :
                self.pages_per_second.append(len(inputs)/(time))
                self.times.append(time)
            loss = self.criterion(preds,labels)
            self.epoch_loss += loss.item()
            self.IoU += binary_iou(convertBinary(preds),labels)
        
    def flops_count(self):
        n_input = len(features) if self.args.dataset == 'simplex' else 2*len(features)
        return get_model_complexity_info(self.model, (n_input, input_height, input_width), as_strings=True,
                                           print_per_layer_stat=False, verbose=False)