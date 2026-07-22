import torch
import torch.nn as nn
import torch.nn.functional as F

#attention module used- CBAM(spatial attention)
#CBAM = Channel Attention + Spatial Attention.
#focuses more on the impoprtant parts of the face in space
class CBAM(nn.Module):
    def __init__(self,channels,reduction= 8, kernel_size=7):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1) #Applies a 2D adaptive max pooling over an input signal composed of several input planes.
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #Applies a 2D adaptive average pooling over an input signal composed of several input planes
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels//reduction, kernel_size=1, bias= False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels//reduction, out_channels=channels, kernel_size=1, bias=False) 
            )
        # Spatial attention
        self.conv_spatial = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=(kernel_size)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        #channel attention
        max_pool_out = self.mlp(self.max_pool(x))
        avg_pool_out = self.mlp(self.avg_pool(x))
        channel_attention= self.sigmoid(max_pool_out+ avg_pool_out)
        x = x* channel_attention
        #spatial attention 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim= True)
        spatial_attention = self.sigmoid(self.conv_spatial(torch.cat([max_out, avg_out], dim =1)))
        x= x* spatial_attention
        return x
    
class ResidualCBAM(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv1= nn.Conv2d(in_channels=channels, out_channels= channels, kernel_size=3, padding=1, bias= False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2= nn.Conv2d(in_channels= channels, out_channels= channels, kernel_size=3, padding=1, bias= False)
        self.bn2= nn.BatchNorm2d(channels)
        self.cbam = CBAM(channels= channels)
    
    def forward(self, x):
        identity = x
        out= self.conv1(x)
        out= self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)
        out = out + identity  # Residual connection
        out = self.act(out)
        return out 

class PreCNN(nn.Module):
    def __init__(self,in_ch= 3, mid_ch=48, num_blocks=6):
        super().__init__()
        self.head =nn.Sequential(
            nn.Conv2d(in_channels= in_ch, out_channels= mid_ch, kernel_size= 3, padding=1, bias =False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace= True)
        )
        self.body = nn.Sequential(*[ResidualCBAM(mid_ch) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=mid_ch, out_channels=in_ch, kernel_size=3, padding=1, bias=True),
            nn.Tanh() #this will minimize the op range between just -1 and 1
        )

    def forward(self,x):
        f = self.head(x)
        f = self.body(f)
        out = self.tail(f)
        return out
