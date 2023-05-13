import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import random

map = {"mask":[0,1], "no-mask":[1,0]}
invTrans = transforms.Compose([ 
            transforms.Normalize(mean = [ -0.485*1/0.229, -0.456*1/0.229, -0.406*1/0.229 ], std = [ 1/0.229, 1/0.224, 1/0.225 ])])


def explore_dataset(dataset, num, show_mask=False, invtrans=False):
    
    # Loop through the images and display them with the bounding boxes
    for i, (image, boxes, labels, mask_face, img_id) in enumerate(dataset):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        if invtrans:
          image = invTrans(image)
        image = image.numpy().transpose(1, 2, 0)
        image = (image*255).astype("uint8")
        mask_face = mask_face.numpy()
        mask_face = (mask_face*255).astype("uint8")
        seg = np.zeros((image.shape[0], image.shape[1], 3))
        if show_mask:
          seg = mask_face[:,:,1:]
        else:
          seg[:,:,:-1] = mask_face[:,:,1:]
        # Convert the tensor image to a numpy array           
        ax.imshow(0.5*image/255 + 0.5*seg/255)
        if i==num:
           break

def count_class_occurance(dataset, class_map):
  print(f'Number of Images in dataset:  {len(dataset)}')

  count = torch.zeros((len(class_map)))
  for image, boxes, labels, mask_face, img_id in dataset:
      count += torch.sum(labels,0)

  classes = list(class_map.keys())

  for i in range(len(classes)):
    if classes[i] == "back":
      continue
    print(f'Number of instances of class "{classes[i]}": {int(count[i])}')
  
  print(f'Number of instances of all classes: {int(count.sum())}')

def plot_train_val(df):
    train_loss = df["train_loss"]
    val_loss = df["val_loss"]
    epochs = len(train_loss)
    best_epoch = np.argmin(df["val_loss"])+1
    fig, axs = plt.subplots(figsize=(10,5))
    plt.plot(np.arange(1, epochs+1), train_loss, label="Training Loss")
    plt.plot(np.arange(1, epochs+1), val_loss, label="Validation Loss")
    plt.title("Train-Val Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xlim([0,epochs+10])
    plt.legend()

    # Plot marker for best epoch
    plt.scatter(best_epoch, val_loss[best_epoch-1], s=100, marker='o', color='red')

    # Plot dotted line from best epoch to x-axis
    plt.plot([best_epoch, best_epoch], [0, val_loss[best_epoch-1]], linestyle='--', color='gray')
    # Plot dotted line from best epoch to y-axis
    plt.plot([0, best_epoch], [val_loss[best_epoch-1], val_loss[best_epoch-1]], linestyle='--', color='gray')
    # Add text
    plt.text(best_epoch, val_loss[best_epoch-1]*1.2, 'Best Model', fontsize = 10, 
         bbox = dict(facecolor = 'red', alpha = 0.5))
    plt.show()


def plot_grid(patches, patch_size, grid_size,seed):
  random.seed(seed)
  random.shuffle(patches)
  grid = np.zeros((patch_size[0]*10, patch_size[1]*10, 3))
  count = 0
  for i in range(grid_size[1]):
      for j in range(grid_size[0]):
          grid[i*patch_size[1]: (i+1)*patch_size[1], j*patch_size[0]:(j+1)*patch_size[0],:] = patches[count]
          count+=1
  fig, axs = plt.subplots(figsize=(20, 20))
  plt.imshow(grid)




