import torchvision
import torch
import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import logging
import torchvision.transforms.v2

from config.config import (features,
                           labels,
                           duplex_labels,
                           img_extension,
                           input_height,
                           input_width,
                           output_height,
                           output_width,
                           pin_memory,
                           detect_labels,
                           detect_duplex_labels,
                           train_test_ratio,
                           train_val_ratio,
                           train_dataset_name,
                           val_dataset_name,
                           test_dataset_name)
from utils.check_data import checkInput, checkLabel

class SimplexDataset(Dataset):
    '''
    This class is for simplex, which each data instance represent a page
    '''
    
    def __init__(self, input_folder,label_folder, transform=None,transform_output=None):
        '''
        This function is the initialization of the dataset object
        
        param input_folder: The filepath to the root folder contains input/channel images
        param label_folder: The filepath to the root folder contains label/mask images
        param transform: Transformation function for input images 
        param transform_output: Transformation function for label images
        
        '''   
        logging.info("Preparing SimplexDataset...")
        
        ## Setting up default transformation if it is None
        if transform == None: 
            self.transform = torchvision.transforms.v2.Resize((input_height,input_width),interpolation=0,antialias=True)
        else:
            self.transform = transform
            
        if transform_output == None: 
            self.transform_output = torchvision.transforms.v2.Resize((output_height,output_width),interpolation=0,antialias=True)
        else:
            self.transform_output = transform_output
        self.trans2Tensor = torchvision.transforms.v2.Compose([torchvision.transforms.v2.ToImage(), torchvision.transforms.v2.ToDtype(torch.float32, scale=True)])
        
        ## Initialization
        self.label_folder = label_folder
        self.input_folder = input_folder
        self.dataset = []
        
        ## For each intermediate result, check if it is valid and then append to the dataset list
        for root,dir,files in os.walk(label_folder):
            for file in files:
                if file == "results.json":
                    _json = open(os.path.join(root,file))
                    _json = json.load(_json)
                    intermediate = _json["intermediate_results"]
                    for page in intermediate:
                        pdf_name = page['pdf_filename'].replace('.pdf','')
                        pn = (4-len(str(page['page_num'])))*"0" + str(page['page_num'])
                        if checkInput(self.input_folder,pdf_name,pn) and checkLabel(page,self.label_folder,pdf_name):
                            self.dataset.append({'page':page,'name':pdf_name})
        logging.info("Finished preparing SimplexDataset")
        logging.info(f'''Total {self.__len__()} samples''')
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        '''
        This function is used to get the data given index of it in the dataset list
        
        param: index of the data in the dataset list
        
        Return:
        Tuplex: Tensor array with number of channels x height x width, Tensor array with number of characteristics x height x width
        '''
        
        ## get the page info
        pn = str(self.dataset[index]['page']['page_num'])
        pdf_name = self.dataset[index]['name']
        page = self.dataset[index]['page']
        pn = (4-len(str(page['page_num'])))*"0" + str(pn)
        
        ## append the page's channel images
        img = []
        for feature in features:
            filenamme = feature+"-"+pn+"-grayscale"+img_extension
            path = os.path.join(self.input_folder,pdf_name,filenamme)
            img.append(self.transform(torchvision.io.read_image(path)/255))
            
        ## append the page's label images
        output = []
        for label in labels:
            output.append(self.transform_output(self.trans2Tensor(Image.open(os.path.join(self.label_folder,pdf_name,page[label])))))
        return tuple((torch.cat(img),torch.cat(output)))
    
class DuplexDataset(Dataset):
    '''
    This class is for duplex , which each data instance represent 2 pages
    '''
    
    def __init__(self, input_folder,label_folder, transform=None,transform_output=None):
        '''
        This function is the initialization of the dataset object
        
        param input_folder: The filepath to the root folder contains input/channel images
        param label_folder: The filepath to the root folder contains label/mask images
        param transform: Transformation function for input images 
        param transform_output: Transformation function for label images
        
        '''   
        logging.info("Preparing DuplexDataset...") 
        
        ## Setting up default transformation if it is None
        if transform == None: 
            self.transform = torchvision.transforms.v2.Resize((input_height,input_width),interpolation=0,antialias=True)
        else:
            self.transform = transform
        if transform_output == None: 
            self.transform_output = torchvision.transforms.v2.Resize((output_height,output_width),interpolation=0,antialias=True)
        else:
            self.transform_output = transform_output
        self.emptyInputTransform = torchvision.transforms.v2.Compose([torchvision.transforms.v2.ToPILImage(),
                         torchvision.transforms.v2.Resize((input_height,input_width)),
                         torchvision.transforms.v2.Compose([torchvision.transforms.v2.ToImage(), torchvision.transforms.v2.ToDtype(torch.float32, scale=True)])])
        self.emptyOutputTransform = torchvision.transforms.v2.Compose([torchvision.transforms.v2.ToPILImage(),
                         torchvision.transforms.v2.Resize((output_height,output_width)),
                         torchvision.transforms.v2.Compose([torchvision.transforms.v2.ToImage(), torchvision.transforms.v2.ToDtype(torch.float32, scale=True)])])
        self.trans2Tensor = torchvision.transforms.v2.Compose([torchvision.transforms.v2.ToImage(), torchvision.transforms.v2.ToDtype(torch.float32, scale=True)])
        self.data_augmentation = [torchvision.transforms.v2.RandomHorizontalFlip(1.0),torchvision.transforms.v2.RandomVerticalFlip(1.0)]
        
        ## Initialization
        self.label_folder = label_folder
        self.input_folder = input_folder
        self.dataset = []
        
        ## For each intermediate result, check if it is valid and then append to the dataset list
        for root,dir,files in os.walk(label_folder):
            for file in files:
                if file == "results.json":
                    _json = open(os.path.join(root,file))
                    _json = json.load(_json)
                    intermediate = _json["intermediate_results"]
                    data = {}
                    for page in intermediate:
                        pdf_name = page['pdf_filename'].replace('.pdf','')
                        pn = (4-len(str(page['page_num'])))*"0" + str(page['page_num'])
                        if checkInput(self.input_folder,pdf_name,pn) and checkLabel(page,self.label_folder,pdf_name):
                            
                            ## When it is odd pages (assume odd pages will be the 'first' page), initialize data variable with first page info 
                            if page['page_num']%2:
                                data = {'page1':page,'name':pdf_name,'transform':0}
                                ## append with only first page info, to create more data with missing second page samples in the dataset
                                self.dataset.append(data)
                                for i in range(1,len(self.data_augmentation)+1):
                                    data['transform'] = i
                                    self.dataset.append(data)
                            
                            ## When it is even pages (assume even pages will be the 'second' page), append second page info to data, and then append to dataset
                            elif data['page1']['page_num']+1 == page['page_num'] and data['name'] == pdf_name:
                                data['page2'] = page
                                data['transform'] = 0
                                self.dataset.append(data)
                                for i in range(1,len(self.data_augmentation)+1):
                                    data['transform'] = i
                                    self.dataset.append(data)
        logging.info("Finished preparing DuplexDataset")
        logging.info(f'''Total {self.__len__()} samples''')
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        '''
        This function is used to get the data given index of it in the dataset list
        
        param: index of the data in the dataset list
        
        Return:
        Tuplex: Tensor array with number of channels x height x width, Tensor array with number of characteristics x height x width
        '''
        
        ## get the page info
        pdf_name = self.dataset[index]['name']
        page1 = self.dataset[index]['page1']
        pn1 = str(self.dataset[index]['page1']['page_num'])
        pn1 = (4-len(str(pn1)))*"0" + str(pn1)
        
        ## append the page's channel images
        img = []
        for feature in features:
            filenamme = feature+"-"+pn1+"-grayscale"+img_extension
            path = os.path.join(self.input_folder,pdf_name,filenamme)
            if self.dataset[index]['transform'] == 0:
                img.append(self.transform(torchvision.io.read_image(path)/255))
            else:
                img.append(self.data_augmentation[self.dataset[index]['transform']-1](self.transform(torchvision.io.read_image(path)/255)))
                
        ## append the page's label images (simplex characteristics only)
        output = []
        for label in labels:
            if self.dataset[index]['transform'] == 0:
                output.append(self.transform_output(self.trans2Tensor(Image.open(os.path.join(self.label_folder,pdf_name,page1[label])))))
            else:
                output.append(self.data_augmentation[self.dataset[index]['transform']-1](self.transform_output(self.trans2Tensor(Image.open(os.path.join(self.label_folder,pdf_name,page1[label]))))))
        
        ## append the page's channel and label images (duplex characteristics)
        if 'page2' in self.dataset[index]:
            pn2 = str(self.dataset[index]['page2']['page_num'])
            page2 = self.dataset[index]['page2']
            pn2 = (4-len(str(pn2)))*"0" + str(pn2)
            for feature in features:
                filenamme = feature+"-"+pn2+"-grayscale"+img_extension
                path = os.path.join(self.input_folder,pdf_name,filenamme)
                if self.dataset[index]['transform'] == 0:
                    img.append(self.transform(torchvision.io.read_image(path)/255))
                else:
                    img.append(self.data_augmentation[self.dataset[index]['transform']-1](self.transform(torchvision.io.read_image(path)/255)))
            for label in duplex_labels:
                if self.dataset[index]['transform'] == 0:
                    output.append(self.transform_output(self.trans2Tensor(Image.open(os.path.join(self.label_folder,pdf_name,page1[label])))))
                else:
                    output.append(self.data_augmentation[self.dataset[index]['transform']-1](self.transform_output(self.trans2Tensor(Image.open(os.path.join(self.label_folder,pdf_name,page1[label]))))))
            for label in labels:
                if self.dataset[index]['transform'] == 0:
                    output.append(self.transform_output(self.trans2Tensor(Image.open(os.path.join(self.label_folder,pdf_name,page2[label])))))
                else:
                    output.append(self.data_augmentation[self.dataset[index]['transform']-1](self.transform_output(self.trans2Tensor(Image.open(os.path.join(self.label_folder,pdf_name,page2[label]))))))
                
        ## When there is no second page, create empty tensor arrays to substitiute as second page
        else:
            for feature in features:
                if self.dataset[index]['transform'] == 0:
                    img.append(self.emptyInputTransform(torch.zeros((1,1)).byte()))
                else:
                    img.append(self.data_augmentation[self.dataset[index]['transform']-1](self.emptyInputTransform(torch.zeros((1,1)).byte())))
            for label in duplex_labels:
                if self.dataset[index]['transform'] == 0:
                    output.append(self.emptyOutputTransform(torch.zeros((1,1)).byte()))
                else:
                    output.append(self.data_augmentation[self.dataset[index]['transform']-1](self.emptyOutputTransform(torch.zeros((1,1)).byte())))
            for label in labels:
                if self.dataset[index]['transform'] == 0:
                    output.append(self.emptyOutputTransform(torch.zeros((1,1)).byte()))
                else:
                    output.append(self.data_augmentation[self.dataset[index]['transform']-1](self.emptyOutputTransform(torch.zeros((1,1)).byte())))
        return tuple((torch.cat(img),torch.cat(output)))

class InputSimplexDataset(Dataset):
    '''
    This class is for simplex, which each data instance represent a page, and only load inputs without ground true images, 
    it is intended to use it to output predicted masked images only
    '''
    def __init__(self,args,transform = None):
        '''
        This function is the initialization of the dataset object
        
        param args: arg object that has 'input_folder'
        param transform: Transformation function for input images 
        '''
        
        ## Setting up default transformation if it is None
        if transform == None: 
            self.transform = torchvision.transforms.v2.Resize((input_height,input_width,),interpolation=0,antialias=True)
        else:
            self.transform = transform
            
        ## Initialization
        items = os.listdir(args.input_folder)
        
        ## Check if all the channels exists before adding them to dataset list 
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
        '''
        This function is used to get the data given index of it in the dataset list
        param: index of the data in the dataset list
        
        Return:
        Tuplex: pdf name, pdf page number, Tensor array with number of channels x height x width
        '''
        
        ## append each channel image
        data = self.dataset[index]
        img = []
        for feature in features:
            img.append(self.transform(torchvision.io.read_image(data[feature])/255))
        return tuple((data['name'],data['pn'],torch.cat(img)))
    
class InputDuplexDataset(Dataset):
    '''
    This class is for duplex, which each data instance represent 2 pages, and only load inputs without ground true images, 
    it is intended to use it to output predicted masked images only
    '''
    
    def __init__(self,args,transform = None):
        '''
        This function is the initialization of the dataset object
        
        param args: arg object that has 'input_folder'
        param transform: Transformation function for input images 
        '''
        
        ## Setting up default transformation if it is None
        if transform == None: 
            self.transform = torchvision.transforms.v2.Resize((input_height,input_width,),interpolation=0,antialias=True)
        else:
            self.transform = transform
        self.emptyInputTransform = torchvision.transforms.v2.Compose([torchvision.transforms.v2.ToPILImage(),
                         torchvision.transforms.v2.Resize((input_height,input_width)),
                         torchvision.transforms.v2.Compose([torchvision.transforms.v2.ToImage(), torchvision.transforms.v2.ToDtype(torch.float32, scale=True)])])
        
        ## Initialization
        items = os.listdir(args.input_folder)
        
        ## Check if all the channels exists before adding them to dataset list 
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
        '''
        This function is used to get the data given index of it in the dataset list

        param: index of the data in the dataset list
        
        Return:
        Tuplex: pdf name, pdf page number, Tensor array with number of channels x height x width
        '''
        data = self.dataset[index]
        img = []
        for feature in features:
            img.append(self.transform(torchvision.io.read_image(data[feature+"1"])/255))
        ## check if second page exists, if yes, append them, if not, append empty tensor arrays.
        if features[0]+"2" in data:
            for feature in features:
                img.append(self.transform(torchvision.io.read_image(data[feature+"2"])/255))
        else:
            for feature in features:
                tmp = self.emptyInputTransform(torch.Tensor(1,1).byte())
                img.append(tmp)
        return tuple((data['name'],data['pn'],torch.cat(img)))

class SimplexDetectDataset(Dataset):
    '''
    This class is for simplex dataset, which each data instance represent a page, and only load inputs without ground true images
    '''
    def __init__(self, input_folder,label_folder,transform=None):   
        '''
        This function is the initialization of the dataset object
        
        param input_folder:  The filepath to the root folder contains input/channel images
        param algorithms:he filepath to the root folder contains label/mask images
        param transform: Transformation function for input images 
        '''
        logging.info("Preparing SimplexDetectDataset...")
        
        ## Setting up default transformation if it is None
        if transform == None: 
            self.transform = torchvision.transforms.v2.Resize((input_height,input_width),interpolation=0,antialias=True)
        else:
            self.transform = transform

        ## Initialization
        self.input_folder = input_folder
        
        ## for each algorithms result, check if it has valid input, then append it to the dataset list
        self.dataset = []
        for root,dir,files in os.walk(label_folder):
            for file in files:
                if file == "results.json":
                    _json = open(os.path.join(root,file))
                    _json = json.load(_json)
                    pdf_name = _json["pdf_filename"]
                    results = _json["algorithm_results"]
                    pdf_name = pdf_name.replace('.pdf','')
                    page_num = 0
                    for _results in results:
                        result = _results['results']
                        page_num += 1
                        data = {'name':pdf_name,'page_num':page_num}
                        for characteristic in result:
                            if characteristic['characteristic'] in detect_labels:
                                pn = (4-len(str(page_num)))*"0" + str(page_num)
                                if checkInput(self.input_folder,pdf_name,pn):
                                    if len(characteristic['boundingBoxResults']):
                                        data[characteristic['characteristic']] = 1
                                    else:
                                        data[characteristic['characteristic']] = 0
                        self.dataset.append(data)
        logging.info("Finished preparing SimplexDetectDataset")
        logging.info(f'''Total {self.__len__()} samples''')
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        '''
        This function is used to get the data given index of it in the dataset list
        
        param: index of the data in the dataset list
        
        Return:
        Tuplex: Tensor array with number of channels x height x width, tensor array with true or false for each characteristic
        '''
        
        ## Initiaiization
        pdf_name = self.dataset[index]['name']
        pn = (4-len(str(self.dataset[index]['page_num'])))*"0" + str(self.dataset[index]['page_num'])
        
        ## Append each channel images for the page
        img = []
        for feature in features:
            filenamme = feature+"-"+pn+"-grayscale"+img_extension
            path = os.path.join(self.input_folder,pdf_name,filenamme)
            img.append(self.transform(torchvision.io.read_image(path)/255))
            
        ## Append true if there at least one bounding boxes, false if there is no bounding boxes for each characteristics
        output = []
        for label in detect_labels:
            output.append(torch.Tensor([self.dataset[index][label]]))
        return tuple((torch.cat(img),torch.cat(output)))
    
    
class DuplexDetectDataset(Dataset):
    '''
    This class is for simplex dataset, which each data instance represent two pages, and only load inputs without ground true images
    '''
    
    def __init__(self, input_folder,label_folder,transform=None):   
        '''
        This function is the initialization of the dataset object
        
        param input_folder:  The filepath to the root folder contains input/channel images
        param algorithms:he filepath to the root folder contains label/mask images
        param transform: Transformation function for input images 
        '''
        logging.info("Preparing DuplexDetectDataset...")
        
        ## Setting up default transformation if it is None
        if transform == None: 
            self.transform = torchvision.transforms.v2.Resize((input_height,input_width),interpolation=0,antialias=True)
        else:
            self.transform = transform
        self.emptyInputTransform = torchvision.transforms.v2.Compose([torchvision.transforms.v2.ToPILImage(),
                         torchvision.transforms.v2.Resize((input_height,input_width)),
                         torchvision.transforms.v2.Compose([torchvision.transforms.v2.ToImage(), torchvision.transforms.v2.ToDtype(torch.float32, scale=True)])])
        self.emptyTensor = torch.Tensor(1,1)
        
        ## Initialization
        self.input_folder = input_folder
        
        ## for every 2 algorithms result, check if it has valid input, then append it to the dataset list
        self.dataset = []
        for root,dir,files in os.walk(label_folder):
            for file in files:
                if file == "results.json":
                    _json = open(os.path.join(root,file))
                    _json = json.load(_json)
                    pdf_name = _json["pdf_filename"]
                    results = _json["algorithm_results"]
                    pdf_name = pdf_name.replace('.pdf','')
                    page_num = 0
                    for _results in results:
                        result = _results['results']
                        page_num += 1
                        _p = "2"
                        if page_num%2:
                            data = {'name':pdf_name,'page_num1':page_num}
                            _p = "1"
                        for characteristic in result:
                            if characteristic['characteristic'] in detect_labels+detect_duplex_labels:
                                pn = (4-len(str(page_num)))*"0" + str(page_num)
                                if checkInput(self.input_folder,pdf_name,pn):
                                    if len(characteristic['boundingBoxResults']):
                                        data[characteristic['characteristic']+_p] = 1
                                    else:
                                        data[characteristic['characteristic']+_p] = 0
                        if not page_num%2:
                            data['page_num2'] = page_num
                            self.dataset.append(data)
        logging.info("Finished preparing DuplexDetectDataset")
        logging.info(f'''Total {self.__len__()} samples''')
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        '''
        This function is used to get the data given index of it in the dataset list
        
        param: index of the data in the dataset list
        
        Return:
        Tuplex: Tensor array with number of channels x height x width, tensor array with true or false for each characterisitc
        '''
        
        ## Initiaiization
        pdf_name = self.dataset[index]['name']
        pn1 = (4-len(str(self.dataset[index]['page_num1'])))*"0" + str(self.dataset[index]['page_num1'])
        pn2 = (4-len(str(self.dataset[index]['page_num2'])))*"0" + str(self.dataset[index]['page_num2'])
        
        ## append each first page channel images
        img = []
        for feature in features:
            filenamme = feature+"-"+pn1+"-grayscale"+img_extension
            path = os.path.join(self.input_folder,pdf_name,filenamme)
            img.append(self.transform(torchvision.io.read_image(path)/255))
            
        ## append each second page channel images
        for feature in features:
            filenamme = feature+"-"+pn2+"-grayscale"+img_extension
            path = os.path.join(self.input_folder,pdf_name,filenamme)
            img.append(self.transform(torchvision.io.read_image(path)/255))
            
        ## Append true if there at least one bounding boxes, false if there is no bounding boxes for each characteristics
        output = []
        for label in detect_labels:
            output.append(torch.Tensor([self.dataset[index][label+'1']]))
        for label in detect_duplex_labels:
            output.append(torch.Tensor([self.dataset[index][label+'1']]))
        for label in detect_labels:
            output.append(torch.Tensor([self.dataset[index][label+'2']]))
        return tuple((torch.cat(img),torch.cat(output)))
    
def collate_fn(batch):
    # Return a tuple instead of a dictionary
    return tuple(torch.stack([x[key] for x in batch]) for key in ['input', 'HighMoistureSimplexPage_bitmap', 'HighMoistureSimplexObject1.3cm_bitmap', 'HighMoistureSimplexObject2.5cm_bitmap'])


def load_dataloader(dataset,data_type,batch_size):
    '''
    This function prepares dataloaders given the dataset
    
    param dataset: the dataset object
    param data_type: the type of dataset object, this is for save file name
    param batch_size: the batch size to load from dataloader
    
    Return:
    train_loader: the dataloader contains the training dataset
    val_loader: the dataloader contains the validation dataset
    test_loader: the dataloader contains the testing dataset
    '''
    logging.info("Preparing Dataloader...")
    
    ## Initialization 
    batch_size = int(batch_size)
    
    ## Create folders for dataloaders if not exists
    if not os.path.exists('./dataloader'):
        os.mkdir('./dataloader')
    path = os.path.join('./dataloader',data_type)
    if not os.path.exists(path):
        os.mkdir(path)
        
    ## If there are no dataloader save files, split dataset and initialize into training dataloader, validation loader, and testing dataloader, then return them
    if not os.path.exists(os.path.join(path,train_dataset_name)) or not os.path.exists(os.path.join(path,val_dataset_name)) or not os.path.exists(os.path.join(path,test_dataset_name)):
        train_set, test_set = random_split(dataset,train_test_ratio,torch.Generator())
        train_set, val_set = random_split(train_set,train_val_ratio,torch.Generator())
        ## save the datasets
        torch.save(train_set,os.path.join(path,train_dataset_name))
        torch.save(val_set,os.path.join(path,val_dataset_name))
        torch.save(test_set,os.path.join(path,test_dataset_name))
    ## If there are dataloader save files, load and return them
    else:
        train_set = torch.load(os.path.join(path,train_dataset_name))
        val_set = torch.load(os.path.join(path,val_dataset_name))
        test_set = torch.load(os.path.join(path,test_dataset_name))
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=pin_memory)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    logging.info("Done preparing Dataloader")
    return train_loader,val_loader,test_loader
