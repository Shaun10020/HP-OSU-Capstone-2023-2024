import torchvision
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import logging

from config.config import features,labels,duplex_labels,img_extension,input_height,input_width,output_height,output_width,pin_memory,random
from utils.check_data import checkInput, checkLabel

class SimplexDataset(Dataset):
    def __init__(self, input_folder,label_folder,intermediate, transform=None,transform_output=None):   
        logging.info("Preparing SimplexDataset...")
        if transform == None: 
            self.transform = torchvision.transforms.Resize((input_height,input_width),antialias=True)
        else:
            self.transform = transform
        if transform_output == None: 
            self.transform_output = torchvision.transforms.Resize((output_height,output_width),antialias=True)
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
        logging.info(f'''Total {self.__len__()} samples''')
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        pn = str(self.dataset[index]['page']['page_num'])
        pdf_name = self.dataset[index]['name']
        page = self.dataset[index]['page']
        pn = (4-len(str(page['page_num'])))*"0" + str(pn)
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
            self.transform = torchvision.transforms.Resize((input_height,input_width),antialias=True)
        else:
            self.transform = transform
        if transform_output == None: 
            self.transform_output = torchvision.transforms.Resize((output_height,output_width),antialias=True)
        else:
            self.transform_output = transform_output
        self.emptyInputTransform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                         torchvision.transforms.Resize((input_height,input_width)),
                         torchvision.transforms.ToTensor()])
        self.emptyOutputTransform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                         torchvision.transforms.Resize((output_height,output_width)),
                         torchvision.transforms.ToTensor()])
        self.trans2Tensor = torchvision.transforms.ToTensor()
        self.label_folder = label_folder
        self.input_folder = input_folder
        self.dataset = []
        for result in intermediate:
            data = {}
            for page in result:
                pdf_name = page['pdf_filename'].replace('.pdf','')
                pn = (4-len(str(page['page_num'])))*"0" + str(page['page_num'])
                if checkInput(self.input_folder,pdf_name,pn) and checkLabel(page,self.label_folder,pdf_name):
                    if page['page_num']%2:
                        data = {'page1':page,'name':pdf_name}
                        self.dataset.append(data)
                    elif data['page1']['page_num']+1 == page['page_num'] and data['name'] == pdf_name:
                        data['page2'] = page
                        self.dataset.append(data)
        logging.info("Finished preparing DuplexDataset")
        logging.info(f'''Total {self.__len__()} samples''')
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        pdf_name = self.dataset[index]['name']
        page1 = self.dataset[index]['page1']
        pn1 = str(self.dataset[index]['page1']['page_num'])
        pn1 = (4-len(str(pn1)))*"0" + str(pn1)
        img = []
        output = []
        for feature in features:
            filenamme = feature+"-"+pn1+"-grayscale"+img_extension
            path = os.path.join(self.input_folder,pdf_name,filenamme)
            img.append(self.transform(torchvision.io.read_image(path)))
        for label in labels:
            output.append(self.transform_output(self.trans2Tensor(Image.open(os.path.join(self.label_folder,pdf_name,page1[label])))))
        
        if 'page2' in self.dataset[index]:
            pn2 = str(self.dataset[index]['page2']['page_num'])
            page2 = self.dataset[index]['page2']
            pn2 = (4-len(str(pn2)))*"0" + str(pn2)
            for feature in features:
                filenamme = feature+"-"+pn2+"-grayscale"+img_extension
                path = os.path.join(self.input_folder,pdf_name,filenamme)
                img.append(self.transform(torchvision.io.read_image(path)))
            for label in duplex_labels:
                output.append(self.transform_output(self.trans2Tensor(Image.open(os.path.join(self.label_folder,pdf_name,page1[label])))))
            for label in labels:
                output.append(self.transform_output(self.trans2Tensor(Image.open(os.path.join(self.label_folder,pdf_name,page2[label])))))
        else:
            for feature in features:
                tmp = self.emptyInputTransform(torch.Tensor(1,1))
                img.append(tmp.byte())
            for label in duplex_labels:
                tmp = self.emptyOutputTransform(torch.Tensor(1,1))
                output.append(tmp.byte())
            for label in labels:
                tmp = self.emptyOutputTransform(torch.Tensor(1,1))
                output.append(tmp.byte())
        return tuple((torch.cat(img),torch.cat(output)))

class InputSimplexDataset(Dataset):
    def __init__(self,args,transform = None):
        if transform == None: 
            self.transform = torchvision.transforms.Resize((input_height,input_width,),antialias=True)
        else:
            self.transform = transform
        items = os.listdir(args.input_folder)
        self.dataset = []
        for item in items:
            path = os.path.join(args.input_folder,item)
            if os.path.isdir(path): 
                pn = 1
                pgnum = (4-len(str(pn)))*"0" + str(pn)
                while checkInput(args.input_folder,item,pgnum):
                    data = {'pn':pn,'name':item}
                    for feature in features:
                        filename = feature+"-"+str(pgnum)+"-grayscale"+img_extension
                        data[feature] = os.path.join(args.input_folder,item,filename)
                    pn += 1
                    pgnum = (4-len(str(pn)))*"0" + str(pn)
                    self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,index):
        data = self.dataset[index]
        img = []
        for feature in features:
            img.append(self.transform(torchvision.io.read_image(data[feature])))
        return tuple((data['name'],data['pn'],torch.cat(img)))
    
### Placeholder, NOT Done yet !!!!
class InputDuplexDataset(Dataset):
    def __init__(self,args,transform = None):
        if transform == None: 
            self.transform = torchvision.transforms.Resize((input_height,input_width,),antialias=True)
        else:
            self.transform = transform
        items = os.listdir(args.input_folder)
        self.emptyInputTransform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                         torchvision.transforms.Resize((input_height,input_width)),
                         torchvision.transforms.ToTensor()])
        self.dataset = []
        for item in items:
            path = os.path.join(args.input_folder,item)
            if os.path.isdir(path): 
                pn = 1
                pgnum = (4-len(str(pn)))*"0" + str(pn)
                while checkInput(args.input_folder,item,pgnum):
                    data = {'pn':pn,'name':item}
                    for feature in features:
                        filename = feature+"-"+str(pgnum)+"-grayscale"+img_extension
                        data[feature+"1"] = os.path.join(args.input_folder,item,filename)
                    pn += 1
                    pgnum = (4-len(str(pn)))*"0" + str(pn)
                    if checkInput(args.input_folder,item,pgnum):
                        for feature in features:
                            filename = feature+"-"+str(pgnum)+"-grayscale"+img_extension
                            data[feature+"2"] = os.path.join(args.input_folder,item,filename)
                    pn += 1
                    pgnum = (4-len(str(pn)))*"0" + str(pn)
                    self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,index):
        data = self.dataset[index]
        img = []
        for feature in features:
            img.append(self.transform(torchvision.io.read_image(data[feature+"1"])))
        if features[0]+"2" in data:
            for feature in features:
                img.append(self.transform(torchvision.io.read_image(data[feature+"2"])))
        else:
            for feature in features:
                tmp = self.emptyInputTransform(torch.Tensor(1,1))
                img.append(tmp.byte())
        return tuple((data['name'],data['pn'],torch.cat(img)))

def collate_fn(batch):
    # Return a tuple instead of a dictionary
    return tuple(torch.stack([x[key] for x in batch]) for key in ['input', 'HighMoistureSimplexPage_bitmap', 'HighMoistureSimplexObject1.3cm_bitmap', 'HighMoistureSimplexObject2.5cm_bitmap'])


def load_dataloader(dataset,batch_size,ratio):
    logging.info("Preparing Dataloader...")
    if random:
        train_set, val_set = random_split(dataset,ratio,torch.Generator())
    else:
        train_size = len(dataset)*ratio[0]
        train_set = torch.utils.data.Subset(dataset,range(int(train_size)))
        val_set = torch.utils.data.Subset(dataset,range(int(train_size),len(dataset)))
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=pin_memory)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    logging.info("Done preparing Dataloader")
    return train_loader,val_loader
