import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from dataloader.dataloader_support import*
import cv2

def extract_nomask_faces(folder, class_map):
    d_train = Objectdetection('data/train', class_map, (256,256), resize=False)
    for image, boxes, labels, image_path in d_train:
        gate = torch.where(labels[:,1]==1)
        labels = labels[gate]
        boxes = boxes[gate]
        boxes = boxes.numpy().astype(int)
        image = image.permute(1,2,0).numpy()
        image = (image*255).astype(np.uint8)
        if len(boxes):
           for i, box in enumerate(boxes):
                xmin,ymin,xmax,ymax = box
                face = image[ymin:ymax, xmin:xmax,:]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                path = folder + image_path[11:-4] + "_" + str(i) + ".jpg"
                cv2.imwrite(path, face)

