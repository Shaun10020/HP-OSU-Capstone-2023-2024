import torchvision
import torch
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

from config.config import features,labels,img_extension,input_height,input_width,output_height,output_width,train_val_ratio,random_seed,input_label,pin_memory,duplex_labels
from utils.check_data import checkInput, checkLabel

class SimplexDataset(Dataset):
    def __init__(self, input,label_folder,intermediate, transform=None,transform_output=None):   
        if transform == None: 
            transform = torchvision.transforms.Resize((input_height,input_width,),antialias=True)
        if transform_output == None: 
            transform_output = torchvision.transforms.Resize((output_height,output_width,),antialias=True)
        trans2Tensor = torchvision.transforms.ToTensor()
        self.dataset = []
        for result in intermediate:
            for page in result:
                pdf_name = page['pdf_filename'].replace('.pdf','')
                pn = str(page['page_num'])
                while len(pn) <4:
                    pn = '0'+pn
                if checkInput(input,pdf_name,pn) and checkLabel(page,label_folder,pdf_name):
                    img = []
                    for feature in features:
                        filenamme = feature+"-"+str(pn)+"-grayscale"+img_extension
                        path = os.path.join(input,pdf_name,filenamme)
                        img.append(transform(torchvision.io.read_image(path)))
                    data = {input_label:torch.cat(img)}
                    for label in labels:
                        data[label] = transform_output(trans2Tensor(Image.fromarray(cv2.imread(os.path.join(label_folder,pdf_name,page[label])))))
                    self.dataset.append(data)
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
    
class DuplexDataset(Dataset):
    def __init__(self, input,label_folder,intermediate, transform=None,transform_output=None):   
        if transform == None: 
            transform = torchvision.transforms.Resize((input_height,input_width,),antialias=True)
        if transform_output == None: 
            transform_output = torchvision.transforms.Resize((output_height,output_width,),antialias=True)
        trans2Tensor = torchvision.transforms.ToTensor()
        self.dataset = []
        for result in intermediate:
            duplex = False
            for page in result:
                pdf_name = page['pdf_filename'].replace('.pdf','')
                _pn = page['page_num']
                pn = str(_pn)
                while len(pn) <4:
                    pn = '0'+pn
                if checkInput(input,pdf_name,pn) and checkLabel(page,label_folder,pdf_name):
                    if duplex:
                        duplex = False
                        if next != _pn:
                            continue
                        for feature in features:
                            filenamme = feature+"-"+str(pn)+"-grayscale"+img_extension
                            path = os.path.join(input,pdf_name,filenamme)
                            img.append(transform(torchvision.io.read_image(path)))
                        data[input_label] = torch.cat(img)
                        for label in labels:
                            data[label+"_2"] = transform_output(trans2Tensor(Image.fromarray(cv2.imread(os.path.join(label_folder,pdf_name,page[label])))))
                        self.dataset.append(data)
                    elif duplex_labels[0] in page:
                        duplex = True
                        next = _pn + 1
                        img = []
                        data = {}
                        for feature in features:
                            filenamme = feature+"-"+str(pn)+"-grayscale"+img_extension
                            path = os.path.join(input,pdf_name,filenamme)
                            img.append(transform(torchvision.io.read_image(path)))
                        for label in labels:
                            data[label+"_1"] = transform_output(trans2Tensor(Image.fromarray(cv2.imread(os.path.join(label_folder,pdf_name,page[label])))))
                        for label in duplex_labels:
                            data[label] = transform_output(trans2Tensor(Image.fromarray(cv2.imread(os.path.join(label_folder,pdf_name,page[label])))))
                    # elif _pn%2:
                    #     duplex = False
                    #     img = []
                    #     for feature in features:
                    #         filenamme = feature+"-"+str(pn)+"-grayscale"+img_extension
                    #         path = os.path.join(input,pdf_name,filenamme)
                    #         img.append(transform(torchvision.io.read_image(path)))
                    #     for _ in features:
                    #         img.append(transform(torch.zeros(1)))
                    #     data = {input_label:torch.cat(img)}
                    #     for label in labels:
                    #         data[label+"_1"] = transform_output(trans2Tensor(Image.fromarray(cv2.imread(os.path.join(label_folder,pdf_name,page[label])))))
                    #     for label in labels:
                    #         data[label+"_2"] = transform_output(torch.zeros(1))
                    #     for label in duplex_labels:
                    #         data[label] = transform_output(torch.zeros(1))
                    #     self.dataset.append(data)
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
    

def load_dataloader(dataset,batch_size):
    train_set, val_set = random_split(dataset,train_val_ratio,torch.Generator().manual_seed(random_seed))
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=pin_memory)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    return train_loader,val_loader