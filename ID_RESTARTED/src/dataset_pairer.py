import os
import glob
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PairDataset(Dataset): #custom  new class inhereting from torch.utils.data.Dataset
    #will produce pairs of images and a label(if same or if different)
    #positives= directory containing positive images(used as anchor)
    #negatives= directory containing negative images (images of other people)
    #pairs_per_epoch: defines the logical length of dataset (how many pairs per epoch).
    #img_size: desired square image size (height and width) after resize.
    def __init__(self, positives, negatives, pairs_per_epoch= 4000, img_size= 224 ):
        #buld a sorted list of all files in positive and negative directory with extensions- jpg, png, jpeg
        self.positive_path= sorted(
            glob.glob(os.path.join(positives,"*jpg"))+
            glob.glob(os.path.join(positives, "*jpeg"))+
            glob.glob(os.path.join(positives, "**png"))
        )
        self.negative_path = sorted(
            glob.glob(os.path.join(negatives,"*jpg"))+
            glob.glob(os.path.join(negatives, "*jpeg"))+
            glob.glob(os.path.join(negatives, "**png"))  
        )
        if len(self.positive_path)<2:
            raise ValueError("Need atleast 2 positive images")
        if len(self.negative_path)<1:
            raise ValueError("Need atleast 1 negative image")
        self.pairs_per_epoch = pairs_per_epoch

        self.tfm = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.RandomHorizontalFlip(p=0.5),#probability of getting flipped
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    def __len__(self):
        return self.pairs_per_epoch
    def _load(self,p):
        img = Image.open(p).convert("RGB")
        return self.tfm(img= img)
    def __getitem__(self, index):
        make_positive= (random.random()<0.5)

        if make_positive:
            a,b= random.sample(self.positive_path, k=2)
            x1= self._load(a)
            x2= self._load(b)
            y = torch.tensor(1.0,dtype= torch.float32)
        else:
            a = random.choice(self.positive_path)
            b= random.choice(self.negative_path)
            x1= self._load(a)
            x2= self._load(b)
            y = torch.tensor(0.0, dtype=torch.float32)
        return x1,x2,y
    