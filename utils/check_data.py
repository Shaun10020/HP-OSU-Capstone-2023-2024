import os
from config.config import features,img_extension,labels,duplex_labels


def checkInput(input,pdf_name,pgnum):
    for feature in features:
        filename = feature+"-"+str(pgnum)+"-grayscale"+img_extension
        path = os.path.join(input,pdf_name,filename)
        if not os.path.isfile(path):
            return False
    return True
    
def checkLabel(page,label_folder,pdf_name):
    for label in labels:
        if not os.path.isfile(os.path.join(label_folder,pdf_name,page[label])):
            return False
    if duplex_labels[0] in page:
        for label in duplex_labels:
            if not os.path.isfile(os.path.join(label_folder,pdf_name,page[label])):
                return False
    return True
        