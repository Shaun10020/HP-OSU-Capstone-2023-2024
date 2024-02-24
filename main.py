
from config.args import get_arguments
from config.config import features,labels,duplex_labels,train_test_ratio,label_extension,threshold,random
from dataloader.load_data import SimplexDataset, DuplexDataset, InputSimplexDataset, InputDuplexDataset, SimplexDetectDataset, DuplexDetectDataset
from utils.load_json import load_results
from utils.save_load_model import load
from utils.convert import convertBinary
from model.UNet import UNet
from model.ENet import ENet
from model.DeepLabV3 import CustomDeepLabV3
from train.train import Train
from train.test import Test

import logging
import os
import torch
import json
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import random_split, DataLoader, Subset

args = get_arguments()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(model,dataset):
    if random:
        train_set, test_set = random_split(dataset,train_test_ratio,torch.Generator())
    else:
        train_size = len(dataset)*train_test_ratio[0]
        train_set = Subset(dataset,range(int(train_size)))
        test_set = Subset(dataset,range(int(train_size),len(dataset)))
    filepath = os.path.join(args.save_folder,f'''{args.model}-{args.dataset}.pt''')
    if os.path.exists(filepath):
        model = load(model,args)
    train = Train(model,device,train_set,args)
    train.run()
    
    model = load(model,args)
    test = Test(model,device,test_set,int(args.batch))
    test.run()
    
    train.save_plot()

def test(model,dataset):
    model = load(model,args)
    test = Test(model,device,dataset,int(args.batch))
    test.run()

def inference(model):
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    model = load(model,args)
    model.eval()
    model.to(device)
    if args.dataset == "simplex":
        data = InputSimplexDataset(args)
    elif args.dataset == "duplex":
        data = InputDuplexDataset(args)
    loader = DataLoader(data,batch_size = int(args.batch))
    
    json_name = ""
    intermediate = []
    for batch in loader:
        input = batch[2].to(device)
        outputs = model(input.float())
        outputs = convertBinary(outputs)
        for name,pn,output in zip(batch[0],batch[1],outputs):
            if not json_name:
                json_name = name
            if json_name != name:
                fp = open(os.path.join(args.output_folder,json_name,"results.json"),"w")
                json.dump({"intermediate_results":intermediate},fp)
                fp.close()
                json_name = name
                intermediate = []
            pn = pn.item()
            pgnum = (4-len(str(pn)))*"0" + str(pn)
            json_instance = {"pdf_filename":name+".pdf","page_num":pn,"intermediate_dir":"intermediate_results/"+pgnum}
            path = os.path.join(args.output_folder,name)
            if not os.path.exists(path):
                os.mkdir(path)
            path = os.path.join(path,"intermediate_results")
            if not os.path.exists(path):
                os.mkdir(path)
            if args.dataset == "simplex":
                path = os.path.join(path,pgnum)
                if not os.path.exists(path):
                    os.mkdir(path)
                for i, label in enumerate(labels):
                    to_pil_image(output[i]).save(os.path.join(path,f'''{label}{label_extension}'''))
                    json_instance[label] = json_instance["intermediate_dir"]+"/"+str(label)+str(label_extension)
                intermediate.append(json_instance)
            
            elif args.dataset == "duplex":
                pgnum2 = (4-len(str(pn)))*"0" + str(pn+1)
                json_instance2 = {"pdf_filename":name+".pdf","page_num":pn+1,"intermediate_dir":"intermediate_results/"+pgnum2}
                path2 = os.path.join(path,pgnum2)
                path = os.path.join(path,pgnum)
                if not os.path.exists(path):
                    os.mkdir(path)
                if not os.path.exists(path2):
                    os.mkdir(path2)
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
                intermediate.append(json_instance)
                intermediate.append(json_instance2)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'''Using device: {device}''')
    
    n_input = len(features) if args.dataset == 'simplex' else 2*len(features)
    n_output = len(labels) if args.dataset == 'simplex' else 2*len(labels)+len(duplex_labels)
    if args.mode != 'inference':
        pdf,algorithm,intermediate = load_results(args.label_folder)
        if args.dataset =='simplex':
            dataset = SimplexDataset(args.input_folder,args.label_folder,intermediate)
        else:
            dataset = DuplexDataset(args.input_folder,args.label_folder,intermediate)
            
    if args.model == 'unet':
        model = UNet(n_input,n_output)
    elif args.model == 'enet':
        model = ENet(n_input,n_output)
    elif args.model == 'deeplabv3':
        model = CustomDeepLabV3(n_output)
    
    if args.mode == 'train':
        train(model,dataset)
    elif args.mode == 'inference':
        inference(model)
    elif args.mode == 'test':
        test(model,dataset)
