import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.arcface import Arcface
from nets.arcface_training import get_lr_scheduler, set_optimizer_lr
from utils.callback import LossHistory
from utils.dataloader import FacenetDataset, LFWDataset, dataset_collate
from utils.utils import (get_num_classes, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    # -------------------------------#
    #   Whether to use CUDA
    #   Set to False if no GPU is available
    # -------------------------------#
    Cuda = True
    # ----------------------------------------------#
    #   Seed: Used to set a random seed
    #   Ensures the same results for independent runs
    # ----------------------------------------------#
    seed = 11
    # ---------------------------------------------------------------------#
    #   distributed: Indicates whether to use multi-GPU distributed training
    #   Terminal commands only support Ubuntu. CUDA_VISIBLE_DEVICES is used to specify GPUs on Ubuntu.
    #   On Windows, DP mode is used by default to access all GPUs; DDP is not supported.
    #   DP mode:
    #       Set            distributed = False
    #       Run command    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP mode:
    #       Set            distributed = True
    #       Run command    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    # ---------------------------------------------------------------------#
    distributed = False
    # ---------------------------------------------------------------------#
    #   sync_bn: Whether to use sync_bn, available in multi-GPU DDP mode
    # ---------------------------------------------------------------------#
    sync_bn = False
    # ---------------------------------------------------------------------#
    #   fp16: Whether to use mixed-precision training
    #   Can reduce memory usage by about half; requires PyTorch 1.7.1 or higher
    # ---------------------------------------------------------------------#
    fp16 = False
    # --------------------------------------------------------#
    #   Path to cls_train.txt in the root directory; contains face image paths and labels
    # --------------------------------------------------------#
    annotation_path = "cls_train.txt"
    # --------------------------------------------------------#
    #   Input image size
    # --------------------------------------------------------#
    input_shape = [112, 112, 3]
    # --------------------------------------------------------#
    #   Backbone feature extraction network options
    #   mobilefacenet, mobilenetv1, iresnet18, iresnet34,
    #   iresnet50, iresnet100, iresnet200
    #
    #   All backbones except mobilenetv1 can train from scratch.
    #   mobilenetv1 is slower to converge due to the lack of residual connections, so it's recommended:
    #   If using mobilenetv1 as backbone, set pretrain = True
    #   If using other backbones, set pretrain = False
    # --------------------------------------------------------#
    backbone = "mobilefacenet"
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   If training was interrupted, set model_path to a weights file in the logs folder to resume training.
    #   Also adjust the training parameters below to ensure continuity of epochs.
    #   
    #   When model_path = '', the entire model is trained from scratch.
    #
    #   To train from the backbone's pretrained weights, set model_path = '', pretrain = True.
    #   To train from scratch, set model_path = '', pretrain = False.
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = ""
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   Whether to use pretrained weights for the backbone. 
    #   Pretrained weights for the backbone are loaded when the model is constructed.
    #   If model_path is set, pretrained weights are ignored.
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained = False

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   Reduce batch_size if there is insufficient GPU memory. 
    #   BatchNorm layers prevent batch_size from being set to 1.
    #
    #   Parameter adjustment recommendations:
    #   (1) Training from pretrained weights:
    #       Adam:
    #           Init_Epoch = 0, Epoch = 100, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0.
    #       SGD:
    #           Init_Epoch = 0, Epoch = 100, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4.
    #       Adjust UnFreeze_Epoch between 100-300.
    #   (2) batch_size:
    #       Set as large as possible within GPU memory limits. If memory is insufficient (OOM or CUDA out of memory),
    #       reduce batch_size. Batch size should be at least 2 due to BatchNorm layers.
    # ----------------------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------#
    #   Training parameters
    #   Init_Epoch: Initial training epoch
    #   Epoch: Total number of epochs
    #   batch_size: Number of images per batch
    # ------------------------------------------------------#
    Init_Epoch = 0
    Epoch = 100
    batch_size = 64

    # ------------------------------------------------------------------#
    #   Other training parameters: learning rate, optimizer, and learning rate decay
    # ------------------------------------------------------------------#
    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01
    optimizer_type = "sgd"
    momentum = 0.9
    weight_decay = 5e-4
    lr_decay_type = "cos"
    save_period = 1
    save_dir = 'logs'
    num_workers = 4
    lfw_eval_flag = True
    lfw_dir_path = "lfw"
    lfw_pairs_path = "model_data/lfw_pair.txt"

    seed_everything(seed)
    # ------------------------------------------------------#
    #   Set GPU devices
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    num_classes = get_num_classes(annotation_path)
    # ---------------------------------#
    #   Load the model and pretrained weights
    # ---------------------------------#
    model = Arcface(num_classes=num_classes, backbone=backbone, pretrained=pretrained)

    if model_path != '':
        # ------------------------------------------------------#
        #   Weight files can be found in the README (Baidu Cloud link provided)
        # ------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   Load weights by matching keys in the model and the pretrained weights
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   Display unmatched keys
        # ------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mNote: It is normal for the head section keys not to load. "
                  "If the backbone section keys fail to load, it is an issue.\033[0m")

    # ----------------------#
    #   Record loss history
    # ----------------------#
    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # ------------------------------------------------------------------#
    #   PyTorch 1.2 does not support AMP; use PyTorch 1.7.1 or above for correct fp16 support
    # ------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    # ----------------------------#
    #   Multi-GPU synchronized BatchNorm
    # ----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not supported with one GPU or without distributed mode.")

    if Cuda:
        if distributed:
            # ----------------------------#
            #   Parallel multi-GPU execution
            # ----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # ---------------------------------#
    #   LFW evaluation
    # ---------------------------------#
    LFW_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=32,
        shuffle=False) if lfw_eval_flag else None

    # -------------------------------------------------------#
    #   0.01 for validation, 0.99 for training
    # -------------------------------------------------------#
    val_split = 0.01
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    show_config(
        num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape, \
        Init_Epoch=Init_Epoch, Epoch=Epoch, batch_size=batch_size, \
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type, \
        save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
    )

    if True:
        # -------------------------------------------------------------------#
        #   Adjust learning rate based on current batch size
        # -------------------------------------------------------------------#
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # ---------------------------------------#
        #   Select optimizer based on optimizer_type
        # ---------------------------------------#
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        # ---------------------------------------#
        #   Get learning rate scheduler
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

        # ---------------------------------------#
        #   Determine epoch lengths
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset is too small for training; please expand the dataset.")

        # ---------------------------------------#
        #   Build dataset loaders
        # ---------------------------------------#
        train_dataset = FacenetDataset(input_shape, lines[:num_train], random=True)
        val_dataset = FacenetDataset(input_shape, lines[num_train:], random=False)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=dataset_collate, sampler=train_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=dataset_collate, sampler=val_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        for epoch in range(Init_Epoch, Epoch):
            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                          Epoch, Cuda, LFW_loader, lfw_eval_flag, fp16, scaler, save_period, save_dir, local_rank)

        if local_rank == 0:
            loss_history.writer.close()