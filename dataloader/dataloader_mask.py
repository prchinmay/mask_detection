import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import glob
import torchvision.transforms as transforms

from utils import*
from metrics import*

trans = transforms.Compose([
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

class Mask_Detection(Dataset):
    def __init__(self, root_dir, class_map, size, transforms=False, data_mine=False):
        self.root_dir = root_dir
        self.annotations = []
        self.map=class_map
        self.width = size[0]
        self.height = size[1]
        self.transforms = transforms
        self.data_mine = data_mine
        for file in os.listdir(root_dir):
            if file.endswith('.xml'):
                annotation_path = os.path.join(root_dir, file)
                image_path = os.path.join(root_dir, file[:-4] + '.jpg')
                self.annotations.append((annotation_path, image_path))
        self.nomask_list = gen_nomask_list("data/nomask_faces",len(self.annotations))
        
    def __len__(self):
        return len(self.annotations)

    def rescale(self, image, boxes):
        # Resize image and annotations
        for box in boxes:
          box[0],box[2] = int(box[0]*self.width/image.size[0]), int(box[2]*self.width/image.size[0])
          box[1],box[3] = int(box[1]*self.height/image.size[1]), int(box[3]*self.height/image.size[1])

        image = TF.resize(image, (self.height, self.width), antialias=True)
    
        return image, boxes
    
    def __getitem__(self, idx):
        annotation_path, image_path = self.annotations[idx]
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # Extract image size
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        
        # Load image
        image = Image.open(image_path)

        # Load Mask for face mask
        mask_mask = load_mask_mask(image_path, self.width, self.height)
        
        # Extract object annotations
        boxes = []
        labels = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.map[label])
  
        # Rescale image and bbox
        image, boxes = self.rescale(image, boxes)

        # Convert annotations to tensors
        image = TF.to_tensor(image) 
        boxes = torch.tensor(boxes, dtype=torch.float32) 
        labels = torch.tensor(labels)
        mask_mask = torch.tensor(mask_mask)

        # Data mining part
        if self.data_mine:
          patches = self.nomask_list[idx]
          image, boxes, labels = pick_nomask(patches, image, boxes, labels)

        # Generate masks of each class and stack them one behind another
        mask_face = bboxes_to_masks(boxes, labels, mask_mask, self.width, self.height)

        if self.transforms:
          image, mask_face = transform(image, mask_face, self.width, self.height)

        return image, boxes, labels, mask_face, image_path[:-4]
    

def collate_fn(batch):
    images = []
    masks_faces = []
    boxes = []
    labels = []
    masks_masks = []
    img_ids = []

    for sample in batch:
        images.append(sample[0])
        boxes.append(sample[1])
        labels.append(sample[2])
        masks_faces.append(sample[3])
        img_ids.append(sample[4])
    
    images = torch.stack(images, dim=0)
    masks_faces = torch.stack(masks_faces, dim=0)
    
    targets = {'boxes':boxes, 'labels':labels, 'masks_faces':masks_faces}
    return images, targets

def bboxes_to_masks(bboxes, labels, mask_mask, width, height):
    """
    Create a segmentation mask with one-hot encoding for each pixel based on the bounding boxes and labels.

    Args:
        bboxes (tensor): bounding boxes of shape (batchsize, N, 4)
        labels (tensor): class labels of shape (batchsize, N)

    Returns:
        tensor: segmentation mask of shape (batchsize, 256, 256, 3)
    """
    batchsize = len(bboxes)
    mask = torch.zeros((height, width, 4), dtype=torch.float32)
    mask[:,:,0] = 1
    mask[:,:,3] = mask_mask[:,:,1]
    for i in range(len(labels)):
        xmin, ymin, xmax, ymax = bboxes[i]
        mask[int(ymin):int(ymax), int(xmin):int(xmax), :-1] = labels[i]
        if torch.abs(labels[i]-torch.tensor([0,0,1])).sum()==.0:
           mask[int(ymin):int(ymax), int(xmin):int(xmax), 2] = mask_mask[int(ymin):int(ymax), int(xmin):int(xmax), 0]
    return mask

def load_mask_mask(image_path, width, height):
    mask_m = np.load(image_path[:-4]+".npy")
    mask_m = cv2.resize(mask_m, (width, height), interpolation = cv2.INTER_AREA)
    _, mask_m = cv2.threshold(mask_m,1,255,cv2.THRESH_BINARY)
    mask_m = (mask_m/255).astype("uint8")

    mask_b = np.logical_not(mask_m)*1

    mask_mask = np.zeros((mask_m.shape[0], mask_m.shape[1], 2))
    mask_mask[:,:,0] = mask_b
    mask_mask[:,:,1] = mask_m
    return mask_mask

def gen_nomask_list(path,n):
    lst1 = np.array(glob.glob(path + "/*"))
    lst2 = np.array(glob.glob(path + "/*"))
    random.seed(20)
    arr1 = np.random.randint(0,len(lst1), n)
    random.seed(0)
    arr2 = np.random.randint(0,len(lst2), n)
    lst1 = lst1[arr1]
    lst2 = lst2[arr2]
    return np.stack((lst1, lst2), axis=1)

def pick_nomask(patches, image, boxes, labels):
    for p in patches:
        randh = random.randint(10, 60)
        randw = int(randh//1.5)
        patch = Image.open(p)
        patch = trans(patch)
        patch = TF.resize(patch, (randh, randw), antialias=True)
        p_h,p_w = patch.shape[1], patch.shape[2]
        i_h, i_w = image.shape[1], image.shape[2]
        x = random.randint(0, i_w - p_w)
        y = random.randint(0, i_h - p_h)
        new_box = [x, y, x+p_w, y+p_h]
        xmin, ymin,xmax, ymax = new_box 
        image[:, ymin:ymax, xmin:xmax] = patch
        boxes = torch.vstack((boxes, torch.tensor(new_box)))
        labels = torch.vstack((labels, torch.tensor([0,1,0])))        
    return image, boxes, labels

def transform(image, mask, width, height):
    #trans to PIL
    to_pil = transforms.ToPILImage()
    mask = mask.permute(2,0,1)
    mask = to_pil(mask)
    image = to_pil(image)


    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(
    image, output_size=(height, width))
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

    # Random horizontal flipping
    if random.random() > 0.5:
       image = TF.hflip(image)
       mask = TF.hflip(mask)

    # Transform to tensor
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)
    return image, mask.permute(1,2,0)
  
    
def Data_Loader_Mask(dataset, batch_size=1, shuffle=True, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)