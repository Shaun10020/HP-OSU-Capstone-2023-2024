import torchvision
import torch
import os
from config.config import features,labels,img_extension,height,width
import cv2
from PIL import Image

def read_input_image(transform,input,pdf_name,pn):
    img = []
    for feature in features:
        filenamme = feature+"-"+str(pn)+"-grayscale"+img_extension
        path = os.path.join(input,pdf_name,filenamme)
        img.append(transform(torchvision.io.read_image(path)))
    return torch.cat(img)
    
def load_dataset(input,label_folder,intermediate):
    transform = torchvision.transforms.Resize((height,width),antialias=True)
    trans2Tensor = torchvision.transforms.ToTensor()
    dataset = []
    for result in intermediate:
        pdf_name = result['pdf_filename'].replace('.pdf','')
        pn = str(result['page_num'])
        while len(pn) <4:
            pn = '0'+pn
        data = {"input":read_input_image(transform,input,pdf_name,pn)}
        for label in labels:
            data[label] = trans2Tensor(Image.fromarray(cv2.imread(os.path.join(label_folder,pdf_name,result[label]))))
        dataset.append(data)
    return dataset
        