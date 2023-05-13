import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from utils import*
import matplotlib.pyplot as plt
import numpy as np



class Objectdetection(Dataset):
    def __init__(self, root_dir, class_map, size, resize=False):
        self.root_dir = root_dir
        self.annotations = []
        self.map=class_map
        self.resize = resize
        self.height = size[0]
        self.width = size[1]       
        for file in os.listdir(root_dir):
            if file.endswith('.xml'):
                annotation_path = os.path.join(root_dir, file)
                image_path = os.path.join(root_dir, file[:-4] + '.jpg')
                self.annotations.append((annotation_path, image_path))
        
    def __len__(self):
        return len(self.annotations)

    def rescale(self, image, boxes):
        # Resize image and annotations
        for box in boxes:
          box[0],box[2] = int(box[0]*self.width/image.size[0]), int(box[2]*self.width/image.size[0])
          box[1],box[3] = int(box[1]*self.height/image.size[1]), int(box[3]*self.height/image.size[1])

        image = TF.resize(image, (self.height, self.width))       
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
            
        # Apply Scaling if specified
        if self.resize:
          image, boxes = self.rescale(image, boxes)

        # Convert annotations to tensors
        image = TF.to_tensor(image)
        boxes = torch.tensor(boxes, dtype=torch.float32) 
        labels = torch.tensor(labels)
        
        return image, boxes, labels, image_path[:-4]
    

def collate_fn(batch):
    images = []
    masks = []
    boxes = []
    labels = []

    for sample in batch:
        images.append(sample[0])
        boxes.append(sample[1])
        labels.append(sample[2])
        masks.append(sample[3])
    
    images = torch.stack(images, dim=0)

    targets = {'boxes':boxes, 'labels':labels}
    return images, targets

  
    
def get_object_detection_data_loader(dataset, batch_size=1, shuffle=True, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)