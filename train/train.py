import logging
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import os
import csv
import torch.nn as nn
import time

from utils.save_load_model import save
from utils.metrics import binary_iou, accuracy, f1_score
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
        self.train_accuracy = []
        self.train_precision = []
        self.train_recall = []
        self.train_f1score = []
        self.IoU = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1score = []
        self.overall = []
        
        ## Setup optimizer and criterion
        if optimizer:
            self.optim = optimizer
        else:
            self.optim = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=1e-5)
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = torch.nn.BCELoss()
        self.lr_updater = torch.optim.lr_scheduler.StepLR(self.optim, int(args.lr_epoch),
                                        float(args.lr_decay))
        logging.info("Done initialize training script")

        
    def run_epoch(self):
        '''
        This function is to run training on the model in one epoch
        '''
        
        ## Initialization, and set the model to train
        self.model.train()
        epoch_loss = 0.0
        IoU = 0.0
        _accuracy = 0.0
        f1score = 0.0
        precision = 0.0
        recall = 0.0
        
        ## Loop through the training dataloader for each batch size
        for batch_data in tqdm(self.train_dataloader):
            self.optim.zero_grad()
            inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
            preds = self.model(inputs.float())
            preds = torch.sigmoid(preds)
            loss = self.criterion(preds,labels)
            loss.backward()
            self.optim.step()
            epoch_loss += loss.item()
            IoU += binary_iou(convertBinary(preds),labels)
            _accuracy += accuracy(convertBinary(preds),labels)
            _precision,_recall, _f1score = f1_score(convertBinary(preds),labels)
            precision += _precision
            recall += _recall
            f1score += _f1score
            
        ## Append training results 
        self.train_IoU.append(IoU / len(self.train_dataloader))
        self.train_accuracy.append(_accuracy / len(self.train_dataloader))
        self.train_precision.append(precision / len(self.train_dataloader))
        self.train_recall.append(recall / len(self.train_dataloader))
        self.train_f1score.append(f1score / len(self.train_dataloader))
        self.epoch_losses.append(epoch_loss / len(self.train_dataloader))
        
    def run_epoch_val(self):
        '''
        This function is to run validation on the model in one epoch
        '''
        
        ## Initialization, and set the model to evaluation
        self.model.eval()
        epoch_loss = 0.0
        IoU = 0.0
        _accuracy = 0.0
        f1score = 0.0
        precision = 0.0
        recall = 0.0
        
        ## Loop through the train dataloader for each batch size
        for batch_data in tqdm(self.val_dataloader):
            inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
            preds = self.model(inputs.float())
            preds = torch.sigmoid(preds)
            loss = self.criterion(preds,labels)
            epoch_loss += loss.item()
            IoU += binary_iou(convertBinary(preds),labels)
            _accuracy += accuracy(convertBinary(preds),labels)
            _precision,_recall, _f1score = f1_score(convertBinary(preds),labels)
            precision += _precision
            recall += _recall
            f1score += _f1score
            
        ## Append validation results 
        self.IoU.append(IoU / len(self.val_dataloader))
        self.accuracy.append(_accuracy / len(self.val_dataloader))
        self.precision.append(precision / len(self.val_dataloader))
        self.recall.append(recall / len(self.val_dataloader))
        self.f1score.append(f1score / len(self.val_dataloader))
        self.epoch_losses_val.append(epoch_loss / len(self.val_dataloader))
            
    def run(self):
        '''
        This function is to run training and validation for the number of epochs
        '''
        logging.info("Running training script...")
        
        ## Loop through each epoch
        for i in range(int(self.args.epoch)):
            logging.info(f'''Running Epoch {i+1}/{int(self.args.epoch)}...''')
            start = time.time()
            self.run_epoch()
            self.lr_updater.step()
            self.run_epoch_val()
            end = time.time()
            self.overall.append(end-start)
            logging.info(f'''Epoch [{i+1}/{int(self.args.epoch)}]
                         Loss: {self.epoch_losses[-1]:.4f}
                         Val Loss: {self.epoch_losses_val[-1]:.4f}
                         Train IoU: {self.train_IoU[-1] * 100:.2f}%
                         Train Accuracy: {self.train_accuracy[-1] * 100:.2f}%
                         Train Precision: {self.train_precision[-1] * 100:.2f}%
                         Train Recall: {self.train_recall[-1] * 100:.2f}%
                         Train F1 Score: {self.train_f1score[-1] * 100:.2f}%
                         IoU: {self.IoU[-1] * 100:.2f}%
                         Accuracy: {self.accuracy[-1] * 100:.2f}%
                         Precision: {self.precision[-1] * 100:.2f}%
                         Recall: {self.recall[-1] * 100:.2f}%
                         F1 Score: {self.f1score[-1] * 100:.2f}%''')
            ## Save the model if it has the lowest validation loss
            if self.epoch_losses_val[-1] == min(self.epoch_losses_val):
                save(self.model,self.args)
        
        ## Save the results in a csv file
        if not os.path.exists("csv_files"):
            os.mkdir("csv_files")
        with open(f'''csv_files/{self.args.model}-{self.args.dataset}-train-epoch.csv''','w',newline='') as fd:
            writer = csv.writer(fd)
            writer.writerow(['Epoch',
                             'Train Loss',
                             'Validation Loss',
                             'Train IoU',
                             'Train Accuracy',
                             'Train Precision',
                             'Train Recall',
                             'Train F1 score',
                             'Validation IoU',
                             'Validation Accuracy',
                             'Validation Precision',
                             'Validation Recall',
                             'Validation F1 score',
                             'Epoch overall time'])
            for i,(train_loss,
                   validation_loss,
                   train_iou,
                   train_accuracy,
                   train_precision,
                   train_recall,
                   train_f1score,
                   validation_iou,
                   validation_accuracy,
                   validation_precision,
                   validation_recall,
                   validation_f1score,
                   overall) in enumerate(zip(self.epoch_losses,
                                             self.epoch_losses_val,
                                             self.train_IoU,
                                             self.train_accuracy,
                                             self.train_precision,
                                             self.train_recall,
                                             self.train_f1score,
                                             self.IoU,
                                             self.accuracy,
                                             self.precision,
                                             self.recall,
                                             self.f1score,
                                             self.overall)):
                writer.writerow([i+1,
                                 train_loss,
                                 validation_loss,
                                 train_iou*100,
                                 train_accuracy*100,
                                 train_precision*100,
                                 train_recall*100,
                                 train_f1score*100,
                                 validation_iou*100,
                                 validation_accuracy*100,
                                 validation_precision*100,
                                 validation_recall*100,
                                 validation_f1score*100,
                                 overall])
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
        
        ## Clear the plot
        plt.clf()
        
        ## Setup titles for Accuracy
        plt.title("Accuracy over Epoch")
        plt.ylabel("Accuracy")
        
        ## Plots the Accuracy, and save as png file
        plt.plot(range(len(self.accuracy)),self.accuracy)
        filename = self.args.model+"-"+self.args.dataset+"-accuracy.png" 
        plt.savefig(os.path.join("plots",filename))
        
        ## Clear the plot
        plt.clf()
        
        ## Setup titles for Precision
        plt.title("Precision over Epoch")
        plt.ylabel("Precision")
        
        ## Plots the Precision, and save as png file
        plt.plot(range(len(self.precision)),self.precision)
        filename = self.args.model+"-"+self.args.dataset+"-precision.png" 
        plt.savefig(os.path.join("plots",filename))
        
        ## Clear the plot
        plt.clf()
        
        ## Setup titles for Recall
        plt.title("Recall over Epoch")
        plt.ylabel("Recall")
        
        ## Plots the Recall, and save as png file
        plt.plot(range(len(self.recall)),self.recall)
        filename = self.args.model+"-"+self.args.dataset+"-recall.png" 
        plt.savefig(os.path.join("plots",filename))
        
        ## Clear the plot
        plt.clf()
        
        ## Setup titles for F1 Score
        plt.title("F1 Score over Epoch")
        plt.ylabel("F1 Score")
        
        ## Plots the F1 Score, and save as png file
        plt.plot(range(len(self.f1score)),self.f1score)
        filename = self.args.model+"-"+self.args.dataset+"-f1score.png" 
        plt.savefig(os.path.join("plots",filename))