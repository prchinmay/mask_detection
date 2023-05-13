import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torchvision.transforms as transforms
import pandas as pd

def compute_iou(box, boxes):
    # Calculate the intersection areas
    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2])
    y1 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
    
    # Calculate the union areas
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    
    # Calculate the IoU (Intersection over Union) between the box and all other boxes
    ious = intersection / union
    
    return ious

def compute_positives(pred_boxs_per_image, true_boxs_per_image, iou_thresh):
    # Sort the predicted boxes by decreasing confidence score

    df_mask = pd.DataFrame(columns=["conf", "tp", "fp"])
    df_nomask = pd.DataFrame(columns=["conf", "tp", "fp"])

    gt_mask_sum = 0
    gt_nomask_sum = 0

    for i in range(len(pred_boxs_per_image)):
        pred_boxes_i = pred_boxs_per_image[i]
        true_boxes_i = true_boxs_per_image[i]
        true_boxes_copy = true_boxes_i.copy()

        for pred_box in pred_boxes_i:
            if len(true_boxes_i)<1:
               break 
            ious = compute_iou(pred_box, true_boxes_i)
            max_idx = np.argmax(ious)

            if ious[max_idx]>=iou_thresh:
               if pred_box[-1]==0:
                  df_nomask.loc[len(df_nomask)] = [pred_box[0], 1, 0]
               else:
                  df_mask.loc[len(df_mask)] = [pred_box[0], 1, 0]
               true_boxes_i = np.delete(true_boxes_i, max_idx, 0)
            else:
                if pred_box[-1]==0:
                  df_nomask.loc[len(df_nomask)] = [pred_box[0], 0, 1]
                else:
                  df_mask.loc[len(df_mask)] = [pred_box[0], 0, 1]

        gt_nomask_sum += len(true_boxes_copy[true_boxes_copy[:, -1]==0])
        gt_mask_sum += len(true_boxes_copy[true_boxes_copy[:, -1]==1])
    
    return  df_mask, df_nomask, gt_mask_sum, gt_nomask_sum

def pr_curve(df_mask, df_nomask, gt_mask_sum, gt_nomask_sum, iou_thresh):

    #Calculate overall metrics
    tp_sum = np.sum(df_mask["tp"]) + np.sum(df_nomask["tp"])
    fp_sum = np.sum(df_mask["fp"]) + np.sum(df_nomask["fp"])
    fn_sum = gt_mask_sum + gt_nomask_sum - tp_sum
    prec = round(tp_sum/(tp_sum + fp_sum), 3)
    recall = round(tp_sum/(tp_sum + fn_sum), 3)
    f1 = round(2*prec*recall/(prec+recall), 3)

    #Calculate metrics for mask
    prec_mask = round(np.sum(df_mask["tp"])/(np.sum(df_mask["tp"]) + np.sum(df_mask["fp"])), 3)
    recall_mask = round(np.sum(df_mask["tp"])/gt_mask_sum, 3)
    f1_mask = round(2*prec_mask*recall_mask/(prec_mask+recall_mask), 3)


    #Plot PR curve for each class
    df_mask = df_mask.sort_values(by=['conf'], ascending=False)
    df_mask["tp"] = np.cumsum(df_mask["tp"])
    df_mask["fp"] = np.cumsum(df_mask["fp"])
    df_mask["fn"] = len(df_mask)*[gt_mask_sum - df_mask.loc[len(df_mask)-1]["tp"]]
    df_mask["precision"] = df_mask["tp"]/(df_mask["tp"]+df_mask["fp"])
    df_mask["recall"] = df_mask["tp"]/(df_mask["tp"]+df_mask["fn"])

    """
    #Plot PR curve for each class
    #df_nomask = df_nomask.sort_values(by=['conf'], ascending=False)
    #df_nomask["tp"] = np.cumsum(df_nomask["tp"])
    #df_nomask["fp"] = np.cumsum(df_nomask["fp"])
    #df_nomask["fn"] = len(df_nomask)*[gt_nomask_sum - df_nomask.loc[len(df_nomask)-1]["tp"]]
    #df_nomask["precision"] = df_nomask["tp"]/(df_nomask["tp"]+df_nomask["fp"])
    #df_nomask["recall"] = df_nomask["tp"]/(df_nomask["tp"]+df_nomask["fn"])
    """

    prec_mask_intrp, recall_mask_intrp, ap_mask = interpolate_precision_recall_curve(df_mask["precision"], df_mask["recall"])
    #prec_nomask_intrp, recall_nomask_intrp, ap_nomask = interpolate_precision_recall_curve(df_nomask["precision"], df_nomask["recall"])

    mAP = round((ap_mask + 0)/2,3)

    fig, axs = plt.subplots(1, 2, figsize=(20,5))
    axs[0].set_title(f'PR Curve @ IOU {iou_thresh} For Mask Detections')
    axs[0].plot(df_mask["recall"], df_mask["precision"], label="Actual")
    axs[0].plot(recall_mask_intrp, prec_mask_intrp, label="Interpolated ")
    axs[0].set_xlim([0,df_mask.iloc[-1]["recall"]+.1])
    axs[0].set_xlabel("Recall")
    axs[0].set_ylabel("Precision")
    axs[0].legend()
    
    print(f'Metrics @ IOU {iou_thresh}')
    print(f'-----------------------------')
    print(f'Class Mask')
    print(f'-----------------------------')
    print(f'Precision : {prec_mask}') 
    print(f'Recall    : {recall_mask}')
    print(f'F1 Score  : {f1_mask}')
    print(f'AP        : {round(ap_mask, 2)}')
    print(f'-----------------------------')

    print(f'Class No-Mask')
    print(f'-----------------------------')
    print(f'Precision : {0}') 
    print(f'Recall    : {0}')
    print(f'F1 Score  : {0}')
    print(f'AP        : {0}')
    print(f'-----------------------------')

    print(f'Overall')
    print(f'-----------------------------')
    print(f'Precision : {prec}') 
    print(f'Recall    : {recall}')
    print(f'F1 Score  : {f1}')
    print(f'mAP       : {mAP}')
    print(f'-----------------------------')
 

def interpolate_precision_recall_curve(precision, recall):
    """
    Interpolates a precision-recall curve with 11 points and calculates the area under the curve.
    :param precision: A list of precision values.
    :param recall: A list of recall values.
    :return: A tuple containing three lists: interpolated_precision, interpolated_recall, and the area under the curve.
    """
    # Check inputs
    assert len(precision) == len(recall)

    # Initialize values for 11-point interpolation
    t = np.linspace(0, 1, 101)
    recall_levels = np.arange(0,1.01,0.01)
    precision_levels = np.zeros((101))

    # Loop over recall levels
    for i, level in enumerate(recall_levels):
        # Find the highest precision at which recall >= current recall level
        precisions_above_recall = np.array([p for p, r in zip(precision, recall) if r >= level])
        if len(precisions_above_recall) == 0:
            precision_levels[i] = 0.0
        else:
            precision_levels[i] = np.max(precisions_above_recall)

    # Calculate area under the curve using trapezoidal rule
    area = np.trapz(precision_levels, recall_levels)

    # Return interpolated precision and recall values, and area under the curve
    return precision_levels, recall_levels, area

def compute_iou_seg(y_true, y_pred):
    """
    Computes the Intersection over Union (IoU) metric between two binary segmentation masks.
    :param y_true: The ground-truth binary segmentation mask.
    :param y_pred: The predicted binary segmentation mask.
    :return: The IoU score between the two masks.
    """
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def evaluate_segmentation_accuracy(y_true, y_pred):
    """
    Evaluates the segmentation accuracy between two sets of binary segmentation masks.
    :param y_true: A list of ground-truth binary segmentation masks.
    :param y_pred: A list of predicted binary segmentation masks.
    :return: The mean Intersection over Union (IoU) score between the two sets of masks.
    """
    iou_scores = []
    for i in range(len(y_true)):
        iou_scores.append(compute_iou_seg(y_true[i], y_pred[i]))
    mean_iou = np.mean(iou_scores)
    print(f'-----------------------------')
    print(f'Metric Face-Mask Segmentation ')
    print(f'-----------------------------')
    print(f'Mean IOU : {round(mean_iou,3)}') 
    print(f'-----------------------------')


