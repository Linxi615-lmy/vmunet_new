from .vmamba import VSSM
import torch
from torch import nn


class VMUNet(nn.Module):
    # def __init__(self,            
    #              input_channels=3, 
    #              num_classes=24,
    #              depths=[2, 2, 9, 2], 
    #              depths_decoder=[2, 9, 2, 2],
    #              drop_path_rate=0.2,
    #              load_ckpt_path=None,
    #             ):
    def __init__(self,                                          
                 input_channels=3,      
                 num_classes=24,
                 depths=[2, 2, 2, 2], 
                 depths_decoder=[2, 2, 2, 1],
                 drop_path_rate=0.2,
                 load_ckpt_path="/20TB/lizewei/Transunet/vssmsmall_dp03_ckpt_epoch_238.pth", 
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                        )
    
    def forward(self, x, epoch=None, iter=None, target=None, valid_mask=None, uarb_cfg=None, force_uarb=False):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        logits = self.vmunet(
            x,
            epoch=epoch,
            iter=iter,
            target=target,
            valid_mask=valid_mask,
            uarb_cfg=uarb_cfg,
            force_uarb=force_uarb,
        )

        # if self.num_classes == 1:
        #     return torch.sigmoid(logits)
        return logits
    
    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_odict = modelCheckpoint['model']

            # 扩展预训练字典：将单一路 encoder 的 keys 复制到 layers_lf 和 layers_hf
            expanded_pretrained = {}
            for k, v in pretrained_odict.items():
                expanded_pretrained[k] = v
                if k.startswith('layers.'):
                    expanded_pretrained[k.replace('layers.', 'layers_lf.')] = v
                    expanded_pretrained[k.replace('layers.', 'layers_hf.')] = v

            # 过滤匹配的 key 并加载到当前模型
            new_dict = {k: v for k, v in expanded_pretrained.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(expanded_pretrained), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            print("encoder loaded finished!")

            # decoder 映射（保持之前的映射逻辑：将 encoder 的层映射到 layers_up）
            model_dict = self.vmunet.state_dict()
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k:
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k:
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k:
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k:
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v

            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            print("decoder loaded finished!")