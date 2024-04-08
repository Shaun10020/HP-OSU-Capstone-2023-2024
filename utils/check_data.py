import os
from config.config import features,img_extension,labels,duplex_labels

def checkInput(input,pdf_name,pgnum):
    '''
    This function check the input fils, based on the image channel. and return True if it is valid, False if it is invalid
    
    param input: the inputs refer the channels for the page to be checked
    param pdf_name: the pdf name to be checked
    param pgnum: the page number of the pdf to be checked
    
    Return:
    Boolean: if the inputs exist or not
    '''
    
    ## For each data, it will check if all the image channel correspond to the entry exists,
    ## if all exists, then return true, else return false.
    for feature in features:
        filename = feature+"-"+str(pgnum)+"-grayscale"+img_extension
        path = os.path.join(input,pdf_name,filename)
        if not os.path.isfile(path):
            return False
    return True
    
def checkLabel(page,label_folder,pdf_name):
    '''
    This function check the label files, based on the file path provided in the json file, and return True if it is valid, False if it is invalid.
    
    param page: a dictionary that contains the page characteristics
    param label_folder: the folder contains the label mask images
    param pdf_name: the pdf name to be checked
    
    Return:
    Boolean: if the labels exist or not
    '''
    
    ## For each data, it will check if the simplex binary masks exists
    for label in labels:
        if not os.path.isfile(os.path.join(label_folder,pdf_name,page[label])):
            return False
        
    ## For each data, if it has duplex label then it will check if the duplex binary masks exists
    if duplex_labels[0] in page:
        for label in duplex_labels:
            if not os.path.isfile(os.path.join(label_folder,pdf_name,page[label])):
                return False
    return True
        