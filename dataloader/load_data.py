import torchvision
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import logging

from config.config import features,labels,img_extension,input_height,input_width,output_height,output_width,random_seed,pin_memory,duplex_labels
from utils.check_data import checkInput, checkLabel

class SimplexDataset(Dataset):
    def __init__(self, input_folder,label_folder,intermediate, transform=None,transform_output=None):   
        logging.info("Preparing SimplexDataset...")
        if transform == None: 
            self.transform = torchvision.transforms.Resize((input_height,input_width,),antialias=True)
        else:
            self.transform = transform
        if transform_output == None: 
            self.transform_output = torchvision.transforms.Resize((output_height,output_width,),antialias=True)
        else:
            self.transform_output = transform_output
        self.trans2Tensor = torchvision.transforms.ToTensor()
        self.label_folder = label_folder
        self.input_folder = input_folder
        self.dataset = []
        for result in intermediate:
            for page in result:
                pdf_name = page['pdf_filename'].replace('.pdf','')
                pn = str(page['page_num'])
                while len(pn) <4:   
                    pn = '0'+pn
                if checkInput(self.input_folder,pdf_name,pn) and checkLabel(page,self.label_folder,pdf_name):
                    self.dataset.append({'page':page,'name':pdf_name})
        logging.info("Finished preparing SimplexDataset")
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        pn = str(self.dataset[index]['page']['page_num'])
        pdf_name = self.dataset[index]['name']
        page = self.dataset[index]['page']
        while len(pn) <4:
            pn = '0'+pn
        img = []
        for feature in features:
            filenamme = feature+"-"+pn+"-grayscale"+img_extension
            path = os.path.join(self.input_folder,pdf_name,filenamme)
            img.append(self.transform(torchvision.io.read_image(path)))
        output = []
        for label in labels:
            output.append(self.transform_output(self.trans2Tensor(Image.open(os.path.join(self.label_folder,pdf_name,page[label])))))
        return tuple((torch.cat(img),torch.cat(output)))
    
class DuplexDataset(Dataset):
    def __init__(self, input_folder,label_folder,intermediate, transform=None,transform_output=None):  
        logging.info("Preparing DuplexDataset...") 
        if transform == None: 
            self.transform = torchvision.transforms.Resize((input_height,input_width,),antialias=True)
        else:
            self.transform = transform
        if transform_output == None: 
            self.transform_output = torchvision.transforms.Resize((output_height,output_width,),antialias=True)
        else:
            self.transform_output = transform_output
        self.trans2Tensor = torchvision.transforms.ToTensor()
        self.label_folder = label_folder
        self.input_folder = input_folder
        self.dataset = []
        duplex = False
        for result in intermediate:
            for page in result:
                pdf_name = page['pdf_filename'].replace('.pdf','')
                pn = str(page['page_num'])
                while len(pn) <4:   
                    pn = '0'+pn
                if checkInput(self.input_folder,pdf_name,pn) and checkLabel(page,self.label_folder,pdf_name):
                    if page['page_num']%2 and duplex_labels[0] in page:
                        data = {'page1':page,'name':pdf_name}
                        duplex = True
                    elif duplex and data['page1']['page_num']+1 == page['page_num'] and data['name'] == pdf_name:
                        duplex = False
                        data['page2'] = page
                        self.dataset.append(data)
        logging.info("Finished preparing DuplexDataset")
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        pn1 = str(self.dataset[index]['page1']['page_num'])
        pn2 = str(self.dataset[index]['page2']['page_num'])
        pdf_name = self.dataset[index]['name']
        page1 = self.dataset[index]['page1']
        page2 = self.dataset[index]['page2']
        while len(pn1) <4:
            pn1 = '0'+pn1
        while len(pn2) <4:
            pn2 = '0'+pn2
        img = []
        for feature in features:
            filenamme = feature+"-"+pn1+"-grayscale"+img_extension
            path = os.path.join(self.input_folder,pdf_name,filenamme)
            img.append(self.transform(torchvision.io.read_image(path)))
        for feature in features:
            filenamme = feature+"-"+pn2+"-grayscale"+img_extension
            path = os.path.join(self.input_folder,pdf_name,filenamme)
            img.append(self.transform(torchvision.io.read_image(path)))
        return
    

def collate_fn(batch):
    # Return a tuple instead of a dictionary
    return tuple(torch.stack([x[key] for x in batch]) for key in ['input', 'HighMoistureSimplexPage_bitmap', 'HighMoistureSimplexObject1.3cm_bitmap', 'HighMoistureSimplexObject2.5cm_bitmap'])


def load_dataloader(dataset,batch_size,ratio):
    logging.info("Preparing Dataloader...")
    train_set, val_set = random_split(dataset,ratio,torch.Generator().manual_seed(random_seed))
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=pin_memory)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    logging.info("Done preparing Dataloader")
    return train_loader,val_loader
