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
        
def checkResult(input,label_folder,intermediate):
    exist = []
    for result in intermediate:
        for page in result:
            pdf_name = page['pdf_filename'].replace('.pdf','')
            pn = str(page['page_num'])
            while len(pn) <4:
                pn = '0'+pn
            if checkInput(input,pdf_name,pn) and checkLabel(page,label_folder,pdf_name):
                exist.append(page)
    return exist