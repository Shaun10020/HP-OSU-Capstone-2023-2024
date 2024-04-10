import logging
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import os
import csv

from utils.save_load_model import save
from utils.metrics import binary_iou
from utils.convert import convertBinary

'''
This script is reponsible for all the model training, and model validation 
'''
class Train:
    
    def __init__(self,
                 model,
                 device,
                 train_dataloader,
                 val_dataloader,
                 args,
                 optimizer = None,
                 criterion = None):
        '''
        This function is the initialization of training class
        
        param model: The model architecture, for example, unet. enet, etc.
        param device: The type of device the training is going to take place, torch.device("cuda") or torch.device("cpu")
        param train_dataloader: the training dataloader, contains the training split of the dataset
        param val_dataloader: the validation dataloader, contains the validation split of the dataset
        param args: the argument parser object, contains model information, dataset information, and training hyperparameters
        param optimizer: the optimizer for the model, default is Adam if it is None
        param criterion: the loss function for the model, default is Binary Cross Entropy Loss if it is None 
        '''
        logging.info("Initializing training script...")
        ## Initialization
        self.device = device
        self.args = args
        self.model = model.to(self.device)
        self.train_dataloader, self.val_dataloader = train_dataloader, val_dataloader
        self.epoch_losses = []
        self.epoch_losses_val = []
        self.train_IoU = []
        self.IoU = []
        
        ## Setup optimizer and criterion
        if optimizer:
            self.optim = optimizer
        else:
            self.optim = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=1e-5)
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = torch.nn.BCELoss()
        self.lr_updater = torch.optim.lr_scheduler.StepLR(self.optim, args.lr_epoch,
                                        args.lr_decay)
        logging.info("Done initialize training script")

        
    def run_epoch(self):
        '''
        This function is to run training on the model in one epoch
        '''
        
        ## Initialization, and set the model to train
        self.model.train()
        epoch_loss = 0.0
        IoU = 0.0
        
        ## Loop through the training dataloader for each batch size
        for batch_data in tqdm(self.train_dataloader):
            self.optim.zero_grad()
            inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
            preds = self.model(inputs.float())
            loss = self.criterion(preds,labels)
            loss.backward()
            self.optim.step()
            epoch_loss += loss.item()
            IoU += binary_iou(convertBinary(preds),labels)
            
        ## Append training results 
        self.train_IoU.append(IoU / len(self.train_dataloader))
        self.epoch_losses.append(epoch_loss / len(self.train_dataloader))
        
    def run_epoch_val(self):
        '''
        This function is to run validation on the model in one epoch
        '''
        
        ## Initialization, and set the model to evaluation
        self.model.eval()
        epoch_loss = 0.0
        IoU = 0.0
        
        ## Loop through the train dataloader for each batch size
        for batch_data in tqdm(self.val_dataloader):
            inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
            preds = self.model(inputs.float())
            loss = self.criterion(preds,labels)
            epoch_loss += loss.item()
            IoU += binary_iou(convertBinary(preds),labels)
            
        ## Append validation results 
        self.IoU.append(IoU / len(self.val_dataloader))
        self.epoch_losses_val.append(epoch_loss / len(self.val_dataloader))
            
    def run(self):
        '''
        This function is to run training and validation for the number of epochs
        '''
        logging.info("Running training script...")
        
        ## Loop through each epoch
        for i in range(int(self.args.epoch)):
            logging.info(f'''Running Epoch {i+1}/{int(self.args.epoch)}...''')
            self.run_epoch()
            self.lr_updater.step()
            self.run_epoch_val()
            logging.info(f'''Epoch [{i+1}/{int(self.args.epoch)}], Loss: {self.epoch_losses[-1]:.4f}, Val Loss: {self.epoch_losses_val[-1]:.4f}, Train IoU: {self.train_IoU[-1] * 100:.2f}%, IoU: {self.IoU[-1] * 100:.2f}%''')
            ## Save the model if it has the lowest validation loss
            if self.epoch_losses_val[-1] == min(self.epoch_losses_val):
                save(self.model,self.args)
        
        ## Save the results in a csv file
        with open(f'''{self.args.model}-{self.args.dataset}-train-epoch.csv''','w') as fd:
            writer = csv.writer(fd)
            writer.writerow(['Epoch','Train Loss','Validation Loss','Train IoU','Validation IoU'])
            for i,(train_loss,validation_loss,train_iou,validation_iou) in enumerate(zip(self.epoch_losses,self.epoch_losses_val,self.train_IoU,self.IoU)):
                writer.writerow([i+1,train_loss,validation_loss,train_iou*100,validation_iou*100])
        logging.info("Done running training script...")
        
    
    def save_plot(self):
        '''
        This function plots and save the results (loss,IoU)) using matplotlib
        '''
        ## Setup titles for loss
        plt.title("Loss over Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        ## Plots the loss results
        plt.plot(range(len(self.epoch_losses)),self.epoch_losses,label = "Training Loss")
        plt.plot(range(len(self.epoch_losses_val)),self.epoch_losses_val,label = "Validation Loss")
        plt.legend()
        
        ## Save as png file
        filename = self.args.model+"-"+self.args.dataset+"-loss.png" 
        if not os.path.exists("plots"):
            os.mkdir("plots")
        plt.savefig(os.path.join("plots",filename))
        
        ## Clear the plot
        plt.clf()
        
        ## Setup titles for IoU
        plt.title("IoU over Epoch")
        plt.ylabel("IoU")
        
        ## Plots the IoU, and save as png file
        plt.plot(range(len(self.IoU)),self.IoU)
        filename = self.args.model+"-"+self.args.dataset+"-IoU.png" 
        plt.savefig(os.path.join("plots",filename))