import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)#上采样
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_size)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)
        return outputs
        
class conv_block(nn.Module):#两个3*3卷积
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(#上采样，卷积，bn,relu
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		        nn.BatchNorm2d(ch_out),
		       	nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
        
class Attention_block(nn.Module): # attention gate for upsampling
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)#编码器
            in_filters = [192, 512, 1024, 3072]#编码器输入128+64，256 + 256,512 + 512，1024+2048
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]#编码器输出
        
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        
        self.Up5 = up_conv(ch_in=2048,ch_out=1024)#第五层特征上采样
        self.Att5 = Attention_block(F_g=512*2,F_l=512*2,F_int=256*2)#第四层注意力
        self.Up_conv5 = conv_block(ch_in=1024*2, ch_out=512*2)#第四层卷积

        self.Up4 = up_conv(ch_in=512*2,ch_out=256*2)#第四层特征上采样
        self.Att4 = Attention_block(F_g=256*2,F_l=256*2,F_int=128*2)#第三层注意力
        self.Up_conv4 = conv_block(ch_in=512*2, ch_out=256*2)#第三层卷积
        
        self.Up3 = up_conv(ch_in=256*2,ch_out=128*2)#第三层特征上采样
        self.Att3 = Attention_block(F_g=128*2,F_l=128*2,F_int=64*2)#第二层注意力
        self.Up_conv3 = conv_block(ch_in=256*2, ch_out=128*2)#第二层卷积
        
        self.Up2 = up_conv(ch_in=128*2,ch_out=64)#第二层上采样
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)#第一层注意力
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)#第一层卷积

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(out_filters[0]),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(out_filters[0]),
                nn.ReLU()
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone
        
    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)#编码器特征64,256,512,1024,2048
        
        '''
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        '''
        #print(feat1.shape, feat2.shape, feat3.shape, feat4.shape, feat5.shape)
        #x = 1/0
        d5 = self.Up5(feat5)#2048->1024
        feat4 = self.Att5(g=d5,x=feat4)#1024
        d5 = torch.cat((feat4,d5),dim=1)#2048        
        d5 = self.Up_conv5(d5)#2048->1024
        
        d4 = self.Up4(d5)
        feat3 = self.Att4(g=d4,x=feat3)
        d4 = torch.cat((feat3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        feat2 = self.Att3(g=d3,x=feat2)
        d3 = torch.cat((feat2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        feat1 = self.Att2(g=d2,x=feat1)
        d2 = torch.cat((feat1,d2),dim=1)
        up1 = self.Up_conv2(d2)
        

        
        
        
        
        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)    # [b, num_classes, h, w]

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
        