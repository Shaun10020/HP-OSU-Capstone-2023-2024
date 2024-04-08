import os
import logging
import torch

def load(model,args):
    '''
    This function load the model's weights
    
    param model: The model structure
    param args: object contain the information, like the model type, model dataset type, model checkpoint filepath 
    
    Return:
    The model with the weights loaded
    '''
    filepath = os.path.join(args.save_folder,f'''{args.model}-{args.dataset}.pt''')
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint)
    return model

def save(model,args):
    '''
    This function save model weight to a folder
    
    param model: The model structure
    param args: object that contain model type, model dataset type, model save folder
    
    '''
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    filepath = os.path.join(args.save_folder,f'''{args.model}-{args.dataset}.pt''')
    logging.info("Saving best model... ["+filepath+"]")
    torch.save(model.state_dict(),filepath)