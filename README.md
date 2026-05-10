# 环境配置:
`E:\论文\医学图像分割\Solo_CXRS31\environment.yml`

注释，需要先把那两个会报错的包从清单里“暂时”注释掉，等环境装好了再手动补装。
```
# - causal-conv1d==1.0.0      
# - mamba-ssm==1.0.1          
# - torch==1.13.0+cu117      
# - torchvision==0.14.0+cu117
# - torchaudio==0.13.0+cu117 
```


## 如果之前的失败环境还在，先删除
conda remove -n vmunet --all -y

## 创建新环境
conda env create -f environment.yml

#激活环境
conda activate vmunet

#安装 PyTorch (带 CUDA 11.7 支持)
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

#验证 GPU  (如果输出 True，继续下一步；如果 False，停下来检查驱动)
python -c "import torch; print(torch.cuda.is_available())"


#最后安装 causal-conv1d 和 mamba-ssm
pip install causal-conv1d==1.0.0
pip install mamba-ssm==1.0.1


## 测试和训练的文件路径
`data_new` 为新的数据集分割测试和训练的文件路径


# 代码使用
`train.py` 为主路径 运行代码从这里运行
注意这里要用绝对路径,这里的路径并不正确要根据实际情况修改

    ```python
    # 没有用到的模型路径
    model_path = '/20TB/lizewei/Vmunet/model_data/imagenet21k+imagenet2012_R50+ViT-B_16.npz'

    # 保存路径
    save_dir = "./output/"
    save_dir = "/media/data/liumengyu/CODE/Solo_CXRS31/output/"

    # 数据路径:即test.txt train.txt val.txt 文件对应的路径,师兄发的
    data_path ="/data_new"
    data_path ="/media/data/liumengyu/CODE/Solo_CXRS31/data_new/"

    # 图像路径: 数据集对应的路径
    image_path = '/xxx/xxx/xxx/xxx'
    image_path = '/media/data/liumengyu/Dataset/MedicalImageDataset/allimg'

    # 模型权重
    load_ckpt_path="/20TB/lizewei/Transunet/vssmsmall_dp03_ckpt_epoch_238.pth"
    load_ckpt_path="/media/data/liumengyu/CODE/Solo_CXRS31/pretrained_models/vssmsmall_dp03_ckpt_epoch_238.pth"
    ```

`utils\dataloader.py` 数据集加载修改

    ```python
    # 原始数据路径
    self.dataset_path = "/20TB/lizewei/allimg/" 
    self.dataset_path = '/media/data/liumengyu/Dataset/MedicalImageDataset/allimg'
    # 高分辨率数据路径
    self.hf_path = "/20TB/lizewei/allimg_H_alpha=0.03/"

    # 掩码路径
    label = cv2.imread(os.path.join("/20TB/lizewei/tmi_31/", str(i), name + '.jpg'), 0) 
    label = cv2.imread(os.path.join('/media/data/liumengyu/Dataset/MedicalImageDataset/tmi_31', str(i), name + '.jpg'), 0) 
    ```