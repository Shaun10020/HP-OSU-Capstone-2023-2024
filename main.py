
from config.args import get_arguments
from config.config import (features,
                           labels,
                           duplex_labels,
                           label_extension)
from dataloader.load_data import (SimplexDataset, 
                                  DuplexDataset, 
                                  InputSimplexDataset, 
                                  InputDuplexDataset, 
                                  load_dataloader)
from utils.save_load_model import load
from utils.convert import convertBinary
from model.UNet import UNet, CustomUNet1
from model.ENet import ENet, CustomENet1
from model.DeepLabV3 import CustomDeepLabV3
from model.DeepLabV3Plus import Deeplabv3Plus
from train.train import Train
from train.test import Test

import logging
import os
import torch
import json
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp


## Get args object, and device for this computer
args = get_arguments()
device = torch.device("cuda") if args.device=="cuda" else torch.device("cpu")
           
def train(model,dataset):
    '''
    This function create train object and intialize training
    
    param model: The mode architecture to be trained
    param dataset: the dataset to be used for training, validation and testing purpose
    '''
    
    ## Setup dataloaders
    train_dataloader,val_dataloader,test_dataloader = load_dataloader(dataset,args.dataset,args.batch)
    
    ## Load model weights to continue training if exist 
    filepath = os.path.join(args.save_folder,f'''{args.model}-{args.dataset}.pt''')
    if os.path.exists(filepath):
        model = load(model,args)
        
    ## Initialize Train class object and start training
    train = Train(model,device,train_dataloader,val_dataloader,args)
    train.run()
    
    ## Initialize Test class object and start testing
    model = load(model,args)
    model = model.module.to(device)
    test = Test(model,device,test_dataloader,args)
    test.run()
    
    ## Save plot 
    train.save_plot()

def test(model,dataset):
    '''
    This function create test object and intialize testing
    
    param model: The mode architecture to be tested
    param dataset: the dataset to be used for testing purpose
    '''
    
    ## Setup dataloaders
    _,_,test_dataloader = load_dataloader(dataset,args.dataset,args.batch)
    
    ## Initialize Test class object and start testing
    model = load(model,args)
    model = model.module.to(device)
    test = Test(model,device,test_dataloader,args)
    test.run()
    
def parallel_test_run(rank,world_size,model,test_dataloader):
    """The target function to run for parallel processes 

    Args:
        rank (int): Integer represent the process id
        world_size (int): Total number of processes
        model: The model architecture
        test_dataloader: The test dataloader to test 
    """
    
    ## Enable logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cpu')
    
    ## Prepare sampler for the process, it will sample different data sample from the dataset depends on the rank
    sampler = DistributedSampler(test_dataloader.dataset,world_size,rank)
    loader_args = dict(batch_size=int(args.batch), num_workers=1, pin_memory=True)
    test_dataloader = DataLoader(test_dataloader.dataset,shuffle=False,sampler=sampler,**loader_args)
    
    ## Initialize and run testing script
    test = Test(model,device,test_dataloader,args,rank=rank)
    test.run()

def parallel_test(model,dataset):
    """The function to initialize and run parallel processes

    Args:
        model: The model architecture to run
        dataset: The dataset to run with 
    """
    
    ## THe function is implemented with CPU in mind
    device = torch.device('cpu')
    
    ## Initialization
    nums = int (os.cpu_count() /2)
    torch.set_num_threads(1)
    processes = []
    mp.set_start_method("spawn")
    
    ## Prepare the model and the data
    model = load(model,args)
    model = model.module.to(device)
    model.share_memory()
    _,_,test_dataloader = load_dataloader(dataset,args.dataset,args.batch)
    
    ## Start processes
    for rank in range(nums):
        p = mp.Process(target=parallel_test_run,args = (rank,nums,model,test_dataloader))
        p.start()
        processes.append(p)
        
    ## Join back processes
    for p in processes:
        p.join()

def inference(model):
    '''
    This function is to run inference on images
    
    param model: the model architecture to be used
    '''
    
    ## Create output folder if not exists
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
        
    ## Load trained model weights and initialization
    model = load(model,args)
    model.eval()
    model.to(device)
    
    ## Prepare data and dataloader
    if args.dataset == "simplex":
        data = InputSimplexDataset(args)
    elif args.dataset == "duplex":
        data = InputDuplexDataset(args)
    loader = DataLoader(data,batch_size = int(args.batch))
    
    
    ## Inference setup
    json_name = ""
    intermediate = []
    
    ## loop through the data
    for batch in loader:
        
        ## predict output from images
        input = batch[2].to(device)
        outputs = model(input.float())
        outputs = torch.sigmoid(outputs)
        outputs = convertBinary(outputs)
        
        ## loop through each output
        for name,pn,output in zip(batch[0],batch[1],outputs):
            ## If have different json file name (indicate new pdf), save the json
            if not json_name:
                json_name = name
            if json_name != name:
                fp = open(os.path.join(args.output_folder,json_name,"results.json"),"w")
                json.dump({"intermediate_results":intermediate},fp)
                fp.close()
                json_name = name
                intermediate = []
                
            ## setup new json instance for this page
            pn = pn.item()
            pgnum = (4-len(str(pn)))*"0" + str(pn)
            json_instance = {"pdf_filename":name+".pdf","page_num":pn,"intermediate_dir":"intermediate_results/"+pgnum}
            
            ## create folders for pdf if not exist
            path = os.path.join(args.output_folder,name)
            if not os.path.exists(path):
                os.mkdir(path)
            path = os.path.join(path,"intermediate_results")
            if not os.path.exists(path):
                os.mkdir(path)
                
            ## If the dataset is simplex
            if args.dataset == "simplex":
                ## create new folder for page if not exist
                path = os.path.join(path,pgnum)
                if not os.path.exists(path):
                    os.mkdir(path)
                    
                ## save the predicted images in the folder
                for i, label in enumerate(labels):
                    to_pil_image(output[i]).save(os.path.join(path,f'''{label}{label_extension}'''))
                    json_instance[label] = json_instance["intermediate_dir"]+"/"+str(label)+str(label_extension)
                
                ## append the json instance
                intermediate.append(json_instance)
            
            ## if the dataset is duplex
            elif args.dataset == "duplex":
                ## setup new json instance for the second page
                pgnum2 = (4-len(str(pn)))*"0" + str(pn+1)
                json_instance2 = {"pdf_filename":name+".pdf","page_num":pn+1,"intermediate_dir":"intermediate_results/"+pgnum2}
                
                ## create new folders for pages if not exist
                path2 = os.path.join(path,pgnum2)
                path = os.path.join(path,pgnum)
                if not os.path.exists(path):
                    os.mkdir(path)
                if not os.path.exists(path2):
                    os.mkdir(path2)
                    
                ## save the predicted images in the folder
                cur = 0
                for i, label in enumerate(labels):
                    to_pil_image(output[cur + i]).save(os.path.join(path,f'''{label}{label_extension}'''))
                    json_instance[label] = json_instance["intermediate_dir"]+"/"+str(label)+str(label_extension)
                cur += len(labels)
                for i, label in enumerate(duplex_labels):
                    to_pil_image(output[cur + i]).save(os.path.join(path,f'''{label}{label_extension}'''))
                    json_instance[label] = json_instance["intermediate_dir"]+"/"+str(label)+str(label_extension)
                cur += len(duplex_labels)
                for i, label in enumerate(labels):
                    to_pil_image(output[cur + i]).save(os.path.join(path2,f'''{label}{label_extension}'''))
                    json_instance2[label] = json_instance2["intermediate_dir"]+"/"+str(label)+str(label_extension)
                    
                ## append the json instances
                intermediate.append(json_instance)
                intermediate.append(json_instance2)


## The main function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'''Using device: {device}''')
    ## Initialize the number of channels for input and output
    n_input = len(features) if args.dataset == 'simplex' else 2*len(features)
    n_output = len(labels) if args.dataset == 'simplex' else 2*len(labels)+len(duplex_labels)
    
    ## Setup dataset
    if args.mode != 'inference':
        if args.dataset =='simplex':
            dataset = SimplexDataset(args.input_folder,args.label_folder)
        else:
            dataset = DuplexDataset(args.input_folder,args.label_folder)
            
    ## Setup model
    if args.model == 'unet':
        model = torch.nn.DataParallel(UNet(n_input,n_output))
    elif args.model == 'enet':
        model = torch.nn.DataParallel(ENet(n_input,n_output))
    elif args.model == 'deeplabv3':
        model = torch.nn.DataParallel(CustomDeepLabV3(n_output))
    elif args.model == 'deeplabv3+':
        model = torch.nn.DataParallel(Deeplabv3Plus(args,n_output))
    elif args.model == 'customunet1':
        model = torch.nn.DataParallel(CustomUNet1(n_input,n_output))
    elif args.model == 'customenet1':
        model = torch.nn.DataParallel(CustomENet1(n_input,n_output))
    
    ## Run the training, testing or inference function
    if args.mode == 'train':
        train(model,dataset)
    elif args.mode == 'inference':
        inference(model)
    elif args.mode == 'test':
        if args.parallel == 'False':
            test(model,dataset)
        else:
            parallel_test(model,dataset)
