import torchvision
import torch
import os
from config.config import features,labels,img_extension,height,width,train_val_ratio,random_seed,batch_size,input_label
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

def read_input_image(transform,input,pdf_name,pn):
    img = []
    for feature in features:
        filenamme = feature+"-"+str(pn)+"-grayscale"+img_extension
        path = os.path.join(input,pdf_name,filenamme)
        img.append(transform(torchvision.io.read_image(path)))
    return torch.cat(img)
        
        
class SimplexDataset(Dataset):
    def __init__(self, input,label_folder,intermediate, transform=None):   
        if transform == None: 
            transform = torchvision.transforms.Resize((height,width),antialias=True)
        trans2Tensor = torchvision.transforms.ToTensor()
        self.dataset = []
        for result in intermediate:
            pdf_name = result['pdf_filename'].replace('.pdf','')
            pn = str(result['page_num'])
            while len(pn) <4:
                pn = '0'+pn
            data = {input_label:read_input_image(transform,input,pdf_name,pn)}
            for label in labels:
                data[label] = transform(trans2Tensor(Image.fromarray(cv2.imread(os.path.join(label_folder,pdf_name,result[label])))))
            self.dataset.append(data)
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
    

def load_dataloader(dataset):
    train_set, val_set = random_split(dataset,train_val_ratio,torch.Generator().manual_seed(random_seed))
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    return train_loader,val_loader