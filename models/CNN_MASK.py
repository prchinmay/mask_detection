import torch.nn as nn
import torch.nn.functional as F

class CNN_MASK(nn.Module):
    def __init__(self):
        super(CNN_MASK, self).__init__()
        
        # Convolutions
        self.conv1 = nn.Conv2d(3, 32, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Deconvolutions
        self.upconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.upconv3 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(16)
        
        #Final convolution
        self.conv_out = nn.Conv2d(16, 4, 3, padding=1)
        self.bn_out = nn.BatchNorm2d(4)

        #drop out
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Downsampling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
                
        # Upsampling
        x = F.relu(self.bn4(self.upconv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.upconv2(x)))
        x = self.dropout(x)
        x = self.bn6(self.upconv3(x))

        # Reducing channels and Sigmoid activation
        x = self.bn_out(self.conv_out(x))
        x = nn.Softmax(dim=1)(x)
        x = x.permute(0, 2, 3, 1)

        return x