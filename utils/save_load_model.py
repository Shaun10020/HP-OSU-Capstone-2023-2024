import os
import logging
import torch

def load(model,args):
    filepath = os.path.join(args.save_folder,f'''{args.model}-{args.dataset}.pt''')
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint)
    return model

def save(model,args):
    filepath = os.path.join(args.save_folder,f'''{args.model}-{args.dataset}.pt''')
    logging.info("Saving best model... ["+filepath+"]")
    torch.save(model.state_dict(),filepath)