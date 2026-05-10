import os
import datetime
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import argparse

from torch.utils.data import DataLoader
# from nets.doubelunet import doubleunet1
# from nets.doubelunet import unet
# from nets.doubelunet import doubleunet
# from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vmunet import VMUNet
# from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
# from networks.vit_seg_modeling_os import VisionTransformer as osViT_seg
# from networks.vit_seg_modeling_os import CONFIGS as osCONFIGS_ViT_seg
# from networks.LViT import LViT as LViT
# from networks.vit_seg_modeling_unet import UNet
# from networks.vit_seg_modeling_unetplus import UnetPulsPuls as UNetPuls
# from networks.vit_seg_modeling_attunet import AttU_Net as AttU_Net
# from networks.vit_seg_modeling_uctransunet import UCTransNet as UCTransNet
# from networks.vit_seg_modeling_unext import UNext_S as UNext
# from networks.Swinunet import SwinTransformerSys as swinunet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch, save_train_visualization


def seed_everything(seed=1234, deterministic=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def select_visualization_batch(dataset, max_search=64):
    search_len = min(len(dataset), max_search)
    for idx in range(search_len):
        sample = dataset[idx]
        if sample is None:
            continue
        if len(sample) == 3:
            _, png, valid = sample
        else:
            _, png = sample
            valid = torch.ones(png.shape[0], dtype=torch.float32)
        if valid.sum() <= 0:
            continue
        if torch.sum(png) <= 0:
            continue
        batch = unet_dataset_collate([sample])
        if batch[0] is not None:
            return batch
    if len(dataset) > 0:
        return unet_dataset_collate([dataset[0]])
    return None


if __name__ == "__main__":
    cuda = True
    # num_classes = 31
    # ICIS
    num_classes = 1
    backbone = "resnet50"
    pretrained = False

    input_shape = (256, 256)
    init_epoch = 0
    freeze_epoch = 0
    unFreeze_epoch = 300
    freeze_batch_size = 2
    unfreeze_batch_size = 8
    freeze_train = False
    init_lr = 1e-4
    min_lr = init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = 'cos'
    eval_flag = True
    eval_period = 1
    num_workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = "/media/data/liumengyu/CODE/UARB_formula/output"
    # data_path = "/media/data/liumengyu/CODE/UARB_formula/data_new"
    # ISIC
    data_path = "/media/data/liumengyu/CODE/UARB_formula/data_new/ISIC2017"
    # image_path = '/media/data/liumengyu/Dataset/MedicalImageDataset/allimg'
    # ISIC
    image_path = '/media/data/liumengyu/Dataset/MedicalImageDataset/ISIC17/allimg'
    loss_func = "BceDice"

    # 三因素联合软权重配置：channel + boundary + current error
    uarb_cfg = {
        # 主损失配置
        "main_loss_name": "bce_dice",
        "stage_loss_name": "bce_dice",
        "main_bce_weight": 1.0,
        "main_dice_weight": 1.0,
        "stage_bce_weight": 1.0,
        "stage_dice_weight": 1.0,
        "tversky_alpha": 0.3,
        "tversky_beta": 0.7,
        "stage_init_weight": 0.1,

        # 基础设置
        #"rib_channels": 24,
        "rib_channels": 1,
        "stage_loss_weights": [0.0, 0.5],   # 先保持你当前只开最高分辨率 stage 的习惯
        "alpha": 0.75,                       # error 在 hard map 中的权重
        "beta": 0.25,                        # uncertainty 在 hard map 中的权重
        "fn_weight": 1.5,
        "fp_weight": 0.5,
        "bg_uncertainty_scale": 0.25,
        "uncertainty_type": 'entropy',

        # 像素级 gate 仍然沿用你原来的聚合方式，但现在聚合的是 boundary_hard
        "pixel_agg_mode": 'max',
        "pixel_agg_blend": 0.7,

        # 新增：边界带参数
        "boundary_radius": 2,

        # 新增：三因素软权重超参数
        "lambda_channel": 0.35,
        "lambda_boundary": 1.50,
        "lambda_error": 0.75,
        "error_gamma": 1.50,
        "weight_cap": 6.0,

        # gate 仍建议先走 pixel，让 UARB 分支重点看边界难点
        "uarb_gate_mode": 'pixel',
        "detach_hard_map": True,

        # 辅助损失 warmup
        "aux_warmup_epochs": 80,
        "aux_max_weight": 0.5,

        # 训练时从哪一种软权重里取图
        "weight_mode": "tri_factor",
    }


    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='../data/Synapse/train_npz', help='root dir for data')
    parser.add_argument('--dataset', type=str, default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int, default=31, help='output channel of network')
    parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int, default=448, help='input patch size of network input')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')


    #  不固定种子
    # args = parser.parse_args()
    # seed_everything(args.seed, bool(args.deterministic))

    # config_vit = CONFIGS_ViT_seg[args.vit_name]
    # config_vit.n_classes = args.num_classes
    # config_vit.n_skip = args.n_skip
    print("##########################################" + backbone + "#######################################################")

    model = VMUNet(
        num_classes=num_classes,
        input_channels=3,
        depths=[2, 2, 2, 2],
        depths_decoder=[2, 2, 2, 1],
        drop_path_rate=0.2,
        load_ckpt_path="/media/data/liumengyu/CODE/UARB_formula/pretrained_models/vssmsmall_dp03_ckpt_epoch_238.pth",
    )
    model.load_from()

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M')
    save_dir = os.path.join(save_dir, 'ModelLabel' + time_str)
    visual_dir = os.path.join(save_dir, 'outputlimian')
    os.makedirs(visual_dir, exist_ok=True)

    loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    model_train = model.train()

    if cuda:
        model_train = torch.nn.DataParallel(model)
        model_train = model_train.cuda()

    with open(os.path.join(data_path, 'train.txt'), 'r') as f:
        train_lines = f.readlines()
    with open(os.path.join(data_path, 'test.txt'), 'r') as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    UnFreeze_flag = False
    batch_size = freeze_batch_size if freeze_train else unfreeze_batch_size

    nbs = 16
    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
    init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
    min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = {
        'adam': optim.Adam(model.parameters(), init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr_fit, min_lr_fit, unFreeze_epoch)

    epoch_step = max(1, num_train // batch_size)
    epoch_step_val = max(1, (num_val + batch_size - 1) // batch_size)

    train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, image_path, ignore_missing_label=True)
    val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, image_path, ignore_missing_label=True)
    vis_dataset = UnetDataset(val_lines, input_shape, num_classes, False, image_path, ignore_missing_label=True)
    vis_batch = select_visualization_batch(vis_dataset)

    train_sampler = None
    val_sampler = None
    eval_callback = EvalCallback(
        model,
        input_shape,
        num_classes,
        val_lines,
        data_path,
        save_dir,
        cuda,
        eval_flag=eval_flag,
        period=eval_period,
    )

    train_generator = torch.Generator()
    # train_generator.manual_seed(args.seed)
    val_generator = torch.Generator()
    # val_generator.manual_seed(args.seed)

    visual_epochs = set(range(0, unFreeze_epoch + 1, 10))

    # 训练前保存 epoch=0 可视化。
    if (0 in visual_epochs) and (vis_batch is not None):
        save_train_visualization(model_train, vis_batch, 0, visual_dir, uarb_cfg=uarb_cfg)

    no_improve_count = 0
    for epoch in range(init_epoch, unFreeze_epoch):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        gen = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=unet_dataset_collate,
            sampler=train_sampler,
            worker_init_fn=seed_worker,
            generator=train_generator,
        )
        gen_val = DataLoader(
            val_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=unet_dataset_collate,
            sampler=val_sampler,
            worker_init_fn=seed_worker,
            generator=val_generator,
        )

        no_improve_count = fit_one_epoch(
            model_train,
            model,
            loss_history,
            eval_callback,
            optimizer,
            epoch,
            epoch_step,
            epoch_step_val,
            gen,
            gen_val,
            unFreeze_epoch,
            loss_func,
            num_classes,
            save_dir,
            no_improve_count,
            uarb_cfg=uarb_cfg,
        )

        current_epoch = epoch + 1
        if (current_epoch in visual_epochs) and (vis_batch is not None):
            save_train_visualization(model_train, vis_batch, current_epoch, visual_dir, uarb_cfg=uarb_cfg)
