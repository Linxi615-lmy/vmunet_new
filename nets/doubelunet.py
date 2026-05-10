import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16
# from resnet import resnet50
# from vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)#上采样
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
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
        self.up = nn.Sequential(
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

class doubleunet1(nn.Module):
    def __init__(self, num_classes=30, pretrained=False, backbone='resnet50'):
        super(doubleunet1, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]
        
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        
        self.Up5 = up_conv(ch_in=1024*2,ch_out=512*2)
        self.Att5 = Attention_block(F_g=512*2,F_l=512*2,F_int=256*2)
        self.Up_conv5 = conv_block(ch_in=1024*2, ch_out=512*2)

        self.Up4 = up_conv(ch_in=512*2,ch_out=256*2)
        self.Att4 = Attention_block(F_g=256*2,F_l=256*2,F_int=128*2)
        self.Up_conv4 = conv_block(ch_in=512*2, ch_out=256*2)
        
        self.Up3 = up_conv(ch_in=256*2,ch_out=128*2)
        self.Att3 = Attention_block(F_g=128*2,F_l=128*2,F_int=64*2)
        self.Up_conv3 = conv_block(ch_in=256*2, ch_out=128*2)
        
        self.Up2 = up_conv(ch_in=128*2,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone
        
    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        
        '''
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        '''
        #print(feat1.shape, feat2.shape, feat3.shape, feat4.shape, feat5.shape)
        #x = 1/0
        d5 = self.Up5(feat5)
        feat4 = self.Att5(g=d5,x=feat4)
        d5 = torch.cat((feat4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
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
        

# class conv_block(nn.Module):
#     def __init__(self,in_ch,out_ch):
#         super(conv_block,self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch,out_ch,kernel_size = 3,stride = 1 ,padding = 1,bias = True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace = True),
#             nn.Conv2d(out_ch,out_ch,kernel_size = 3, stride = 1, padding = 1, bias = True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace = True)
#             )
#     def forward(self,x):
#         return self.conv(x)

# class up_conv1(nn.Module):
#     def __init__(self,in_ch,out_ch):
#         super(up_conv1,self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor = 2),
#             nn.Conv2d(in_ch,out_ch,kernel_size = 3, stride = 1, padding = 1,bias = True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace = True)
#         )
#     def forward(self,x):
#         return self.up(x) 

class unet(nn.Module):
    def __init__(self,in_ch = 3,out_ch = 30):
        super(unet,self).__init__()
        n1 = 64
        filters = [n1,n1*2,n1*4,n1*8,n1*16]#unet是5层网络，每一层的输出通道数
        self.Maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)#四次降采样
        self.Maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv1 = conv_block(in_ch,filters[0])#5层网络，每一层网络的卷积
        self.conv2 = conv_block(filters[0],filters[1])
        self.conv3 = conv_block(filters[1],filters[2])
        self.conv4 = conv_block(filters[2],filters[3])
        self.conv5 = conv_block(filters[3],filters[4])
        self.up5 = up_conv(filters[4],filters[3])#上采样和卷积
        self.up_conv5 = conv_block(filters[4],filters[3])#解码器每一层的卷积
        self.up4 = up_conv(filters[3],filters[2])
        self.up_conv4 = conv_block(filters[3],filters[2])
        self.up3 = up_conv(filters[2],filters[1])
        self.up_conv3 = conv_block(filters[2],filters[1])
        self.up2 = up_conv(filters[1],filters[0])
        self.up_conv2 = conv_block(filters[1],filters[0])
        self.conv = nn.Conv2d(filters[0],out_ch,kernel_size = 1,stride = 1,padding =0)#输出概率图
        #self.active = torch.sigmoid()#概率图
    def forward(self,x):
        e1 = self.conv1(x)#提取特征
        e2 = self.Maxpool1(e1)
        e2 = self.conv2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.conv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.conv4(e4)
        e5 = self.Maxpool4(e4)
        e5 = self.conv5(e5)
        d5 = self.up5(e5)#解码器特征
        d5 = torch.cat((e4,d5),dim = 1)
        d5 = self.up_conv5(d5)
        d4 = self.up4(d5)
        d4 = torch.cat((e3,d4),dim = 1)
        d4 = self.up_conv4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat((e2,d3),dim = 1)
        d3 = self.up_conv3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat((e1,d2),dim = 1)
        d2 = self.up_conv2(d2)
        out = self.conv(d2)
        #out = self.activate(out)
        return out

class doubleunet(nn.Module):
    def __init__(self,in_ch=3,out_ch=24):
        super(doubleunet,self).__init__()
        self.doubleunet1 = doubleunet1(num_classes=30, pretrained=False, backbone='resnet50')
        self.doubleunet2 = unet(30,30)
        self.frazee = False
    def forward(self,x):
        out = self.doubleunet1(x)
        if not self.frazee:
            return out 
        out = self.doubleunet2(out)
        return out
    def freeze(self):
        self.frazee = True
        for param in self.doubleunet1.parameters():
            param.requires_grad = False

        
# if __name__ == '__main__':
#     model = doubleunet(3,30)
#     input = torch.randn(1, 3, 448, 448)  
#     print(model(input).shape)