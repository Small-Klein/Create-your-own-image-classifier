'''python predict.py /path/to/image checkpoint
python predict.py input checkpoint --top_k 3
python predict.py input checkpoint --category_names cat_to_name.json 
python predict.py input checkpoint --gpu'''

import glob, os
from PIL import Image
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import torch 
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    if args.arch == "VGG":
        
        model = models.vgg16(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False
     
        classifier = nn.Sequential(OrderedDict([
    ("fc1", nn.Linear(25088, 4096)),
    ("relu1", nn.ReLU()),
    ("dropout1", nn.Dropout(p=0.5)),
    ("fc2", nn.Linear(4096, 1024)),
    ("relu2", nn.ReLU()),
    ("dropout2", nn.Dropout(p=0.5)),
    ("fc3", nn.Linear(1024, 102)),
    ("logsoftmax", nn.LogSoftmax(dim=1))
]))
                              
    elif args.arch == "Densenet":
        model = models.densenet121(pretrained=True)
        #parameters to be trained, features are frozen
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
             ("fc1", nn.Linear(25088, 4096)),
             ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(p=0.5)),
            ("fc2", nn.Linear(4096, 1024)),
            ("relu2", nn.ReLU()),
            ("dropout2", nn.Dropout(p=0.5)),
            ("fc3", nn.Linear(1024, 102)),
            ("logsoftmax", nn.LogSoftmax(dim=1))
]))                  
                              
                                  
    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    

    return model


def process_image(image):
    pil_image = Image.open(image)
    image_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    np_image = np.array(pil_image)
    np_image = np.transpose(np_image, (2, 0, 1))
    
    
    return np_image





def predict(image_path, model, device, topk=5):
        
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
                                  
    if args.gpu:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.to(device)

    output = model(image)
    with torch.no_grad():
        ps = torch.exp(output)
        top_ps, top_classes = ps.topk(dim=1)

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_labels = [idx_to_class[i] for i in top_classes]
    
    for i in range(len(top_ps)):
        print("{}".format(i + 1), "{}".format(top_labels[i].upper()), "{:.3f}% ".format(top_ps[i] * 100))

    return top_ps, top_labels
#arguments 
parser = argparse.ArgumentParser()
parser.add_argument("image_path", help ="This is the file path of the image we want to classify")
parser.add_argument("--dir", help = "This is the path to the directory")
parser.add_argument("--topk", help = "These are the top matched classes that will be returned", default = 5)
parser.add_argument("--category_names", help = "This is the path that links the cateegory to the real name", default = "cat_to name.json")
parser.add_argument("--gpu", help = " This is when you want to train the model on the GPU", default = False, action = "store_true")


#parse results
args = parser.parse_args()