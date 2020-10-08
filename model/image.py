import random
import os
from PIL import Image
import numpy as np
import h5py
import cv2
import torch
from torchvision import transforms

def load_data(img_path,train = True, direct = False, code = 1):
    transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) ])

    transform2 = transforms.ToTensor()
    gt_path = img_path.replace('.png','.h5').replace("img1",'target'+code).replace('.jpg','.h5')

    for root, dir, filenames in os.walk(img_path):
        if train is True:
            aug = 0
        else:
            aug = 0
        for i in range(len(filenames)):
            if i == 0:
                img =  np.array(Image.open(img_path + "/" + filenames[i]).convert('RGB'))
                if direct is True:
                    return transform2(img)
                img = transform(img)
                image = img
                if aug == 1:
                    image = img.transpose(Image.FLIP_LEFT_RIGHT)
                if aug == 2:
                    crop_size = (int(image.shape[1]/2),int(image.shape[2]/2))
                    if random.randint(0,9)<= 3:
                        dx = int(random.randint(0,1)*image.shape[1]*1./2)
                        dy = int(random.randint(0,1)*image.shape[2]*1./2)
                    else:
                        dx = int(random.random()*image.shape[1]*1./2)
                        dy = int(random.random()*image.shape[2]*1./2)
                    image = image[:,dx:crop_size[0]+dx,dy:crop_size[1]+dy]
                if aug == 0:
                    image = image.unsqueeze(dim = 1)
            else:
                new_img =  np.array(Image.open(img_path + "/" + filenames[i]).convert('RGB'))
                new_img = transform(new_img)
                new_image = new_img
                if aug ==1:
                    new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT)
                if aug == 2:
                    new_image = new_image[:,dx:crop_size[0]+dx,dy:crop_size[1]+dy]
                    new_image = new_image.unsqueeze(dim = 1)
                if aug == 0:
                    new_image = new_image.unsqueeze(dim = 1)
                image = torch.cat([image, new_image], axis = 1)

    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    if aug == 1:
        target = np.fliplr(target)
    if aug == 2:
        target = target[dx:crop_size[0]+dx,dy:crop_size[1]+dy]
    target = cv2.resize(target,(int(np.floor(image.shape[3]/8)), int(np.floor(image.shape[2]/8))),interpolation = cv2.INTER_CUBIC)*64
    return image,target
