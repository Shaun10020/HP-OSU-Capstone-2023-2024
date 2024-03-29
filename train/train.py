import logging
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import os
import csv

from config.config import target
from utils.save_load_model import save
from utils.metrics import binary_iou
from utils.convert import convertBinary

class Train:
    
    def __init__(self,
                 model,
                 device,
                 train_dataloader,
                 val_dataloader,
                 args,
                 optimizer = None,
                 criterion = None):
        logging.info("Initializing training script...")
        self.device = device
        self.args = args
        self.model = model.to(self.device)
        self.train_dataloader, self.val_dataloader = train_dataloader, val_dataloader
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
        self.epoch_losses = []
        self.epoch_losses_val = []
        self.train_IoU = []
        self.IoU = []
        logging.info("Done initialize training script")

        
    def run_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        IoU = 0.0
        for batch_data in tqdm(self.train_dataloader):
            self.optim.zero_grad()
            inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
            preds = self.model(inputs.float())
            loss = self.criterion(preds,labels)
            loss.backward()
            self.optim.step()
            epoch_loss += loss.item()
            IoU += binary_iou(convertBinary(preds),labels)
        self.train_IoU.append(IoU / len(self.train_dataloader))
        self.epoch_losses.append(epoch_loss / len(self.train_dataloader))
        
    def run_epoch_val(self):
        self.model.eval()
        epoch_loss = 0.0
        IoU = 0.0
        for batch_data in tqdm(self.val_dataloader):
            inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
            preds = self.model(inputs.float())
            loss = self.criterion(preds,labels)
            epoch_loss += loss.item()
            IoU += binary_iou(convertBinary(preds),labels)
        self.IoU.append(IoU / len(self.val_dataloader))
        self.epoch_losses_val.append(epoch_loss / len(self.val_dataloader))
            
    def run(self):
        logging.info("Running training script...")
        if target:
            epoch = 1
            logging.info(f'''Running Epoch {epoch}...''')
            self.run_epoch()
            self.lr_updater.step()
            self.run_epoch_val()
            logging.info(f'''Epoch [{epoch}], Loss: {self.epoch_losses[-1]:.4f}, Val Loss: {self.epoch_losses_val[-1]:.4f}, Train IoU: {self.train_IoU[-1] * 100:.2f}%, IoU: {self.IoU[-1] * 100:.2f}%''')
            epoch += 1
            if self.epoch_losses_val[-1] == min(self.epoch_losses_val):
                save(self.model,self.args)
            while self.IoU[-1] * 100 < 95.0:
                logging.info(f'''Running Epoch {epoch}...''')
                self.run_epoch()
                self.lr_updater.step()
                self.run_epoch_val()
                logging.info(f'''Epoch [{epoch}], Loss: {self.epoch_losses[-1]:.4f}, Val Loss: {self.epoch_losses_val[-1]:.4f}, Train IoU: {self.train_IoU[-1] * 100:.2f}%, IoU: {self.IoU[-1] * 100:.2f}%''')
                epoch += 1
                if self.epoch_losses_val[-1] == min(self.epoch_losses_val):
                    save(self.model,self.args)
        else:
            for i in range(int(self.args.epoch)):
                logging.info(f'''Running Epoch {i+1}/{int(self.args.epoch)}...''')
                self.run_epoch()
                self.lr_updater.step()
                self.run_epoch_val()
                logging.info(f'''Epoch [{i+1}/{int(self.args.epoch)}], Loss: {self.epoch_losses[-1]:.4f}, Val Loss: {self.epoch_losses_val[-1]:.4f}, Train IoU: {self.train_IoU[-1] * 100:.2f}%, IoU: {self.IoU[-1] * 100:.2f}%''')
                if self.epoch_losses_val[-1] == min(self.epoch_losses_val):
                    save(self.model,self.args)
        with open(f'''{self.args.model}-{self.args.dataset}-train-epoch.csv''','w') as fd:
            writer = csv.writer(fd)
            writer.writerow(['Epoch','Train Loss','Validation Loss','Train IoU','Validation IoU'])
            for i,(train_loss,validation_loss,train_iou,validation_iou) in enumerate(zip(self.epoch_losses,self.epoch_losses_val,self.train_IoU,self.IoU)):
                writer.writerow([i+1,train_loss,validation_loss,train_iou*100,validation_iou*100])
        logging.info("Done running training script...")
        
    
    def save_plot(self):
        plt.title("Loss over Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        plt.plot(range(len(self.epoch_losses)),self.epoch_losses,label = "Training Loss")
        plt.plot(range(len(self.epoch_losses_val)),self.epoch_losses_val,label = "Validation Loss")
        plt.legend()
        
        filename = self.args.model+"-"+self.args.dataset+"-loss.png" 
        if not os.path.exists("plots"):
            os.mkdir("plots")
        plt.savefig(os.path.join("plots",filename))
        
        plt.clf()
        
        plt.title("IoU over Epoch")
        plt.ylabel("IoU")
        
        plt.plot(range(len(self.IoU)),self.IoU)
        filename = self.args.model+"-"+self.args.dataset+"-IoU.png" 
        plt.savefig(os.path.join("plots",filename))