import cv2
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb

def gen_mask_gt(dataset, light_skin, dark_skin):
  patches = []
  patch_size = (150,150)
  for image, boxes, labels, img_id in dataset:
    gate = torch.where(labels[:,2]==1)
    labels = labels[gate]
    boxes = boxes[gate]
    boxes = boxes.numpy().astype(int)
    image = image.permute(1,2,0).numpy()
    image = (image*255).astype(np.uint8)
  
    full_mask, patches = make_patches(image, boxes, patches, patch_size, light_skin, dark_skin)
    out_path = img_id + ".npy"
    np.save(out_path, full_mask)

def make_patches(image, boxes, patches, patch_size, light_skin, dark_skin):
    full_mask = np.zeros((image.shape[0], image.shape[1]), dtype="uint8") 
    for i, box in enumerate(boxes):
        
        xmin,ymin,xmax,ymax = box
        face = image[ymin:ymax, xmin:xmax,:]
        w,h = face.shape[1], face.shape[0]
        if w<10 and h<10:
           continue
        face_r = cv2.resize(face, patch_size, interpolation = cv2.INTER_AREA)
        r_head = face_r.shape[0]//2 - int(face_r.shape[0]*0)

        hsv_face = cv2.cvtColor(face_r, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_face, light_skin, dark_skin)
        mask = cv2.bitwise_not(mask)

        mask=cv2.ellipse(mask, center=(75, 75), 
                 axes=(100,110), angle=0, startAngle=0, 
                 endAngle=360, color=(0), thickness=75)
   
        mask[:r_head,:] = 0

        kernel_er = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel_er, iterations=1)

        edges = cv2.dilate(cv2.Canny(mask,0,255),None)
        cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)
        if len(cnt)<2:
           continue
        cnt = cnt[-1]
        dummy = np.zeros(patch_size, np.uint8)
        mask = cv2.drawContours(dummy, [cnt],-1, 255, -1)

        mask3d = np.zeros((patch_size[0],patch_size[1],3))
        mask3d[:,:,0] = mask
        patch = 0.6*(face_r/255) + 0.4*(mask3d/255)
        patches.append(patch)

        mask = cv2.resize(mask, (w,h), interpolation = cv2.INTER_AREA)
        full_mask[ymin:ymax, xmin:xmax] = mask
        
    return full_mask, patches