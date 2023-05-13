import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

invTrans = transforms.Compose([ 
            transforms.Normalize(mean = [ -0.485*1/0.229, -0.456*1/0.229, -0.406*1/0.229 ], std = [ 1/0.229, 1/0.224, 1/0.225 ])])

def get_boxes_pred(contours, pred, cls, thresh=0.25):
  boxes = []
  area = []        
  for contour in contours:
      rect = cv2.boundingRect(contour)
      x,y,w,h = rect
      prob = np.mean(pred[y:y+h, x:x+w, cls])
      area.append(w*h)
      boxes.append([prob, x, y, x+w, y+h, cls])
  area, boxes = np.array(area), np.array(boxes)
  if len(area)>0:
    return boxes[area>(np.max(area)*thresh)]
  else:
    return boxes[area>(np.mean(area)*thresh)]

def get_boxes_true(boxes, labels):
    boxes = np.concatenate((torch.ones((len(boxes),1)), boxes), axis=1)
    boxes = np.concatenate((boxes, labels[:,-1].reshape(-1,1)), axis=1)
    return boxes

def make_mask_seg(pred_face):
    mask_seg = pred_face[:,:,3]
    mask_seg[mask_seg<0.5] = 0
    mask_seg[mask_seg>=0.5] = 1
    plot_mask = np.zeros((pred_face.shape[0], pred_face.shape[1], 3)).astype('uint8')
    plot_mask[:,:,-1] = mask_seg
    return mask_seg, plot_mask


def process_pred(pred_face, cls, thresh):
    mask_b = pred_face[:,:,cls]
    mask_b[mask_b < 0.7] = 0
    mask_b[mask_b >= 0.7] = 1
    
    mask_b = (mask_b*255).astype("uint8")
    ret,mask_b = cv2.threshold(mask_b,50,255,cv2.THRESH_BINARY)

    mask_b = (mask_b*255).astype("uint8")
    ret,mask_b = cv2.threshold(mask_b,50,255,cv2.THRESH_BINARY)

    #get predicted bboxes
    contours = sorted(cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)
    return get_boxes_pred(contours, pred_face, cls, thresh)

def test_mask(model, test_data, path, thresh, show, inv_trans=False):

    device = "cpu"
    new_class_map = {0:(255,0,0), 1: (0,0,255)}
    true_boxs_per_image = []
    pred_boxs_per_image = []
    p_mask_seg = []
    t_mask_seg = []
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)

    # Apply the model to the test images
    model.eval()
    with torch.no_grad():
        for batch_idx, (x_test, y_test) in enumerate(test_data):
            x_test = x_test.to(device)
            gt_boxes, gt_labels, gt_masks = y_test["boxes"], y_test["labels"], y_test["masks_faces"]
            gt_boxes, gt_labels, gt_masks  = gt_boxes[0].numpy(), gt_labels[0].numpy(), gt_masks[0].numpy()

            #predict
            pred_face = model(x_test)
            if inv_trans:
               x_test = invTrans(torch.tensor(x_test))
        
            pred_face, x_test = pred_face.cpu(), x_test.cpu().permute(0,2,3,1)
            pred_face, x_test = np.squeeze(np.array(pred_face)), np.squeeze(np.array(x_test))
            pred_face_slice, x_test = pred_face[:, :, 1:3], np.ascontiguousarray(x_test*255, dtype=np.uint8)
        
            boxes_for_img = []
            for cls in range(len(new_class_map)):
                boxes_pred = process_pred(pred_face_slice, cls, thresh)
          
                #plot bboxes 
                for box in boxes_pred:
                    boxes_for_img.append(box)
                    prob, xmin, ymin, xmax, ymax, cls = box
                    if show["bboxes"]:
                      cv2.rectangle(x_test,(int(xmin),int(ymin)),(int(xmax),int(ymax)),new_class_map[cls],2)

            true_boxs_per_image.append(get_boxes_true(gt_boxes, gt_labels))
            pred_boxs_per_image.append(boxes_for_img)
            face_mask, plot_mask = make_mask_seg(pred_face)
            p_mask_seg.append(face_mask)
            t_mask_seg.append(gt_masks[:,:,-1])
            x_test = cv2.cvtColor(x_test, cv2.COLOR_BGR2RGB)

            if show["images"]:
                if show["mask_seg"]:
                    cv2_imshow(0.6*x_test+0.4*plot_mask*255)

                if show["face_seg"]:
                    pred_face[:, :, 0] = 0
                    pred_face[pred_face<0.7] = 0
                    pred_face[pred_face>=0.7] = 1
                    cv2_imshow(0.6*x_test+0.4*pred_face*255)   
      

    return pred_boxs_per_image, true_boxs_per_image, p_mask_seg, t_mask_seg