# Mask Detection

<center>
  <img src=https://drive.google.com/uc?export=view&id=19TcTNVctD3HHk4mph-pm55TQk867GIIg width="1000"/>
</center>


Check out the image above. These are predictions of a Convolutional Neural Network that predicts faces wearing masks 
and segments the face-mask area. Do you want to know how this is done? This Jupyter Notebook contains a tutorial for 
Face Mask detection in people, using a custom Convolutional Neural Network. The network, dataset class, utility functions 
and train/test scripts are written from scratch in PyTorch and explained in detail. If you have dabbled in Computer Vision/ML/DL 
or just interested in such things, this is for you. 

## Objective
The objective of the project is to train a object detection network that detects people wearing or not wearing a mask. 
Also, it is desired that a segmentation of the mask be extracted from faces wearing the mask. The extraction of face-mask 
segmentations are firstly, performed using traditional OpenCV techniques and later with our custom CNN. 

## Sections

To address the above mentioned objectives, we address 3 sections primarily:

1.   Object detection of mask/no-mask faces using custom CNN
2.   Segmenting face-mask using OpenCV
3.   Segmenting face-mask using our Custom CNN

## Some Results
<center>
  <img src=https://drive.google.com/uc?export=view&id=19lxGkqFcUEIQVDAJtJc3R9VXpUdlbVpv width="1000"/>
</center>
<center>
  <img src=https://drive.google.com/uc?export=view&id=19mDcSjnAcIH-Nsns7vxufAgND35lbrz0 width="1000"/>
</center>

Check out the Jupyter notebook `PEA.ipynb` for the detailed tutorial!
