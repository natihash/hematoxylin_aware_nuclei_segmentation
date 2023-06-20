import torch
import torchvision.transforms.functional as TF
import cv2
import random
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

class MyRotationTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

transform_train = transforms.Compose([transforms.RandomVerticalFlip(),MyRotationTransform([0, 90, 180, 270]),
                                     transforms.RandomHorizontalFlip()])

class CustomImageDataset(Dataset):
    def __init__(self, images, markers, semantics, hemas, weights, transforms):
        self.images = images
        self.markers = markers
        self.semantics = semantics
        self.hemas = hemas
        self.weights = weights
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])/255;img = torch.from_numpy(img);img = img.permute(2, 0, 1)
        hema = cv2.imread(self.hemas[idx], 0)/255;hema = torch.from_numpy(hema)
        sema = cv2.imread(self.semantics[idx], 0)/255;sema = torch.from_numpy(sema)
        mark = cv2.imread(self.markers[idx], 0)/255;mark = torch.from_numpy(mark)+sema
        weight = 1.0*cv2.imread(self.weights[idx], 0)
        weight[weight==0] = 0.3;weight[weight==120] = 0.6;weight[weight==240] = 1.1;weight = torch.from_numpy(weight)
        
        xy = torch.cat([img, hema.unsqueeze(0), sema.unsqueeze(0), mark.unsqueeze(0),
                       weight.unsqueeze(0)], dim=0)
        xy = self.transforms(xy)
        
        return xy[:4], xy[4], xy[5], xy[6]