# -------------------------------------#
#       Train the dataset
# -------------------------------------#
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (Loss, ModelEMA, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import (download_weights, get_classes, seed_everything,
                         show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch

'''
When training your own object detection model, pay attention to the following:
1. Check whether your format meets the requirements before training. This library requires the dataset to be in VOC format. 
   Prepared content includes input images and labels.
   Input images should be .jpg files and do not need to have a fixed size. They will be automatically resized before training.
   Grayscale images will automatically be converted to RGB format for training, no need for manual adjustments.
   If input images are not .jpg, batch convert them to .jpg before starting training.

   Labels should be in .xml format, containing the target information to be detected, corresponding to the input image files.

2. The size of the loss value is used to determine convergence. The trend of convergence is more important, 
   i.e., the validation loss continues to decrease. If the validation loss does not change, the model has likely converged.
   The specific size of the loss value has no absolute meaning. Large or small values depend on the loss calculation method. 
   A lower value does not necessarily indicate better performance. For a better-looking loss value, you can divide the corresponding loss function by 10,000.
   The loss values during training are saved in the logs folder under loss_%Y_%m_%d_%H_%M_%S.

3. Trained weight files are saved in the logs folder. Each training epoch contains several training steps, and gradient descent occurs at each step.
   If you train only a few steps, the weights won't be saved. Understand the concepts of Epoch and Step clearly.
'''

if __name__ == "__main__":
    # ---------------------------------#
    #   Cuda    Whether to use Cuda
    #           Set to False if no GPU is available
    # ---------------------------------#
    Cuda = True
    # ----------------------------------------------#
    #   Seed    Used to fix the random seed
    #           Ensures consistent results across independent training runs
    # ----------------------------------------------#
    seed = 11
    # ---------------------------------------------------------------------#
    #   distributed     Indicates whether to use single-machine multi-GPU distributed training
    #                   Terminal commands are only supported on Ubuntu. CUDA_VISIBLE_DEVICES specifies the GPUs on Ubuntu.
    #                   Windows systems default to DP mode, using all GPUs and do not support DDP.
    #   DP mode:
    #       Set distributed = False
    #       Run command: CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP mode:
    #       Set distributed = True
    #       Run command: CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    # ---------------------------------------------------------------------#
    distributed = False
    # ---------------------------------------------------------------------#
    #   sync_bn     Whether to use sync_bn, available in DDP mode
    # ---------------------------------------------------------------------#
    sync_bn = False
    # ---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training
    #               Reduces memory usage by about half, requires PyTorch 1.7.1 or above
    # ---------------------------------------------------------------------#
    fp16 = True
    # ---------------------------------------------------------------------#
    #   classes_path    Points to the txt file under model_data related to your dataset
    #                   Before training, ensure that classes_path matches your dataset
    # ---------------------------------------------------------------------#
    classes_path = 'target.txt'
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   See README for instructions on downloading weight files. Pre-trained model weights are generalizable across datasets as features are universal.
    #   The critical part of the pre-trained weights is the backbone feature extraction network, used for feature extraction.
    #   Pre-trained weights are recommended in 99% of cases. Without them, the backbone weights are too random, resulting in poor feature extraction and training outcomes.
    #
    #   For interrupted training, set model_path to a weight file from the logs folder to reload partially trained weights.
    #   Adjust the Freeze or Unfreeze parameters to maintain training epoch continuity.
    #
    #   If model_path = '', the model weights are not loaded.
    #
    #   Here, the entire model's weights are loaded during training in train.py.
    #   To train from scratch, set model_path = '', Freeze_Train = False, and train from zero without freezing the backbone.
    #
    #   Training from scratch usually yields poor results due to random weights and weak feature extraction. It is strongly discouraged!
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = r'E:\graduation project1\YOLOV8-master\yolov8\logs\last_epoch_weights.pth'
    # model_path = r'E:\graduation project1\model\yolov8_n.pth'
    # ------------------------------------------------------#
    #   input_shape     Input shape size, must be a multiple of 32
    # ------------------------------------------------------#
    input_shape = [256, 256]
    # ------------------------------------------------------#
    #   phi             YOLOv8 version to use
    #                   n : yolov8_n
    #                   s : yolov8_s
    #                   m : yolov8_m
    #                   l : yolov8_l
    #                   x : yolov8_x
    # ------------------------------------------------------#
    phi = 'n'
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      Whether to use pre-trained weights for the backbone network. Loaded during model construction.
    #                   If model_path is set, backbone weights do not need to be loaded, and pretrained becomes irrelevant.
    #                   If model_path is not set, pretrained = True only loads the backbone for training.
    #                   If model_path is not set, pretrained = False, and Freeze_Train = False, training starts from scratch without freezing the backbone.
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained = True
    # ------------------------------------------------------------------#
    #   mosaic              Mosaic data augmentation.
    #   mosaic_prob         Probability of using mosaic data augmentation at each step, default is 50%.
    #
    #   mixup               Whether to use mixup data augmentation, only valid when mosaic=True.
    #                       Mixup is applied only to images enhanced by mosaic.
    #   mixup_prob          Probability of using mixup data augmentation after mosaic, default is 50%.
    #                       Overall mixup probability is mosaic_prob * mixup_prob.
    #
    #   special_aug_ratio   Reference to YoloX. Mosaic-generated training images significantly deviate 
    #                       from the real distribution of natural images.
    #                       When mosaic=True, this script enables mosaic within the special_aug_ratio range.
    #                       Default is the first 70% of epochs; for 100 epochs, mosaic is enabled for 70.
    # ------------------------------------------------------------------#
    mosaic = True
    mosaic_prob = 0.2
    mixup = True
    mixup_prob = 0.2
    special_aug_ratio = 0.7
    # ------------------------------------------------------------------#
    #   label_smoothing     Label smoothing. Typically set below 0.01, e.g., 0.01 or 0.005.
    # ------------------------------------------------------------------#
    label_smoothing = 0.01

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   Training consists of two phases: freezing and unfreezing. The freezing phase is designed for users with limited hardware performance.
    #   Freezing training requires less memory. On very low-end GPUs, you can set Freeze_Epoch equal to UnFreeze_Epoch, 
    #   Freeze_Train = True, to only perform freezing training.
    #      
    #   Below are some parameter recommendations. Adjust based on your specific needs:
    #   (1) Training from pre-trained weights of the entire model:
    #       Adam:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (freezing)
    #           Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (not freezing)
    #       SGD:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 300, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (freezing)
    #           Init_Epoch = 0, UnFreeze_Epoch = 300, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (not freezing)
    #           UnFreeze_Epoch can be adjusted between 100-300.
    #   (2) Training from scratch:
    #       Init_Epoch = 0, UnFreeze_Epoch >= 300, Unfreeze_batch_size >= 16, Freeze_Train = False (no freezing training).
    #       UnFreeze_Epoch should be at least 300. optimizer_type = 'sgd', Init_lr = 1e-2, mosaic = True.
    #   (3) Setting batch_size:
    #       Choose as large as possible within the GPU's capacity. Batch size affects BatchNorm layers, so it cannot be set to 1, and the minimum is 2.
    #       Normally, Freeze_batch_size should be 1-2 times the size of Unfreeze_batch_size. Avoid large differences, as they affect the automatic learning rate adjustment.
    # ----------------------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Freezing phase training parameters
    #   During this phase, the backbone of the model is frozen, and the feature extraction network does not change.
    #   This requires less memory and fine-tunes the network.
    #   Init_Epoch          Starting training epoch. Its value can exceed Freeze_Epoch to skip the freezing phase
    #                       and directly start from Init_Epoch, adjusting the learning rate accordingly.
    #                       (Used for resuming interrupted training)
    #   Freeze_Epoch        Number of epochs for freezing training
    #                       (Not applicable if Freeze_Train=False)
    #   Freeze_batch_size   Batch size during freezing training
    #                       (Not applicable if Freeze_Train=False)
    # ------------------------------------------------------------------#
    Init_Epoch = 30
    Freeze_Epoch = 30
    Freeze_batch_size = 64
    # ------------------------------------------------------------------#
    #   Unfreezing phase training parameters
    #   During this phase, the backbone of the model is not frozen, and the feature extraction network changes.
    #   This requires more memory, and all parameters of the network are updated.
    #   UnFreeze_Epoch          Total training epochs.
    #                           SGD requires more time to converge, so a larger UnFreeze_Epoch is recommended.
    #                           Adam can use relatively fewer epochs.
    #   Unfreeze_batch_size     Batch size after unfreezing.
    # ------------------------------------------------------------------#
    UnFreeze_Epoch = 60
    Unfreeze_batch_size = 64
    # ------------------------------------------------------------------#
    #   Freeze_Train    Whether to perform freezing training.
    #                   Default is to first freeze the backbone for training and then unfreeze it.
    # ------------------------------------------------------------------#
    Freeze_Train = True

    # ------------------------------------------------------------------#
    #   Other training parameters: learning rate, optimizer, learning rate decay, etc.
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         Maximum learning rate of the model
    #   Min_lr          Minimum learning rate of the model, default is 0.001 of Init_lr
    # ------------------------------------------------------------------#
    Init_lr = 0.001
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  Type of optimizer, options are adam, sgd.
    #                   For Adam optimizer, set Init_lr=1e-3
    #                   For SGD optimizer, set Init_lr=1e-2
    #   momentum        Momentum parameter used within the optimizer
    #   weight_decay    Weight decay to prevent overfitting
    #                   Adam can cause weight_decay errors, so set weight_decay=0 for Adam.
    # ------------------------------------------------------------------#
    optimizer_type = "adam"
    momentum = 0.937
    weight_decay = 0
    # ------------------------------------------------------------------#
    #   lr_decay_type   Learning rate decay strategy, options are step, cos
    # ------------------------------------------------------------------#
    lr_decay_type = "cos"
    # ------------------------------------------------------------------#
    #   save_period     Number of epochs after which weights are saved
    # ------------------------------------------------------------------#
    save_period = 5
    # ------------------------------------------------------------------#
    #   save_dir        Directory to save weights and log files
    # ------------------------------------------------------------------#
    save_dir = 'logs'
    # ------------------------------------------------------------------#
    #   eval_flag       Whether to perform evaluation during training, using the validation set.
    #                   Install pycocotools for better evaluation experience.
    #   eval_period     Frequency of evaluation in epochs. Frequent evaluation is not recommended
    #                   as it is time-consuming and slows training significantly.
    #                   The mAP obtained here differs from get_map.py due to:
    #                   (1) mAP here is calculated on the validation set.
    #                   (2) Conservative evaluation parameters are used to speed up evaluation.
    # ------------------------------------------------------------------#
    eval_flag = True
    eval_period = 5
    # ------------------------------------------------------------------#
    #   num_workers     Number of threads used for data loading
    #                   Increasing this value speeds up data loading but uses more memory.
    #                   For low-memory systems, set it to 2 or 0.
    # ------------------------------------------------------------------#
    num_workers = 4

    # ------------------------------------------------------#
    #   train_annotation_path   Path to training images and labels
    #   val_annotation_path     Path to validation images and labels
    # ------------------------------------------------------#
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'
    seed_everything(seed)
    # ------------------------------------------------------#
    #   Set the GPU(s) to be used
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("GPU Device Count:", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    # ------------------------------------------------------#
    #   Get classes and anchors
    # ------------------------------------------------------#
    # [0] [1,0] [0,1]
    class_names, num_classes = get_classes(classes_path)

    # ----------------------------------------------------#
    #   Download pre-trained weights
    # ----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(phi)
            dist.barrier()
        else:
            download_weights(phi)

    # ------------------------------------------------------#
    #   Create YOLO model
    # ------------------------------------------------------#
    model = YoloBody(input_shape, num_classes, phi, pretrained=pretrained)

    if model_path != '':
        # ------------------------------------------------------#
        #   Load weight file, see README for download instructions
        # ------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   Load weights by matching pre-trained keys with model keys
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
        #   Display keys that were not successfully matched
        # ------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key Num:", len(no_load_key))
            print("\n\033[1;33;44mNote: It is normal for the head part to not load; it is an error if the backbone part does not load.\033[0m")

    # ----------------------#
    #   Define loss function
    # ----------------------#
    yolo_loss = Loss(model)
    # ----------------------#
    #   Log the loss values
    # ----------------------#
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # ------------------------------------------------------------------#
    #   PyTorch 1.2 does not support AMP (Automatic Mixed Precision), 
    #   Torch 1.7.1 or above is recommended for fp16 support.
    # ------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    # ----------------------------#
    #   Sync BatchNorm for multiple GPUs
    # ----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not supported for single GPU or non-distributed training.")

    if Cuda:
        if distributed:
            # ----------------------------#
            #   Run parallel on multiple GPUs
            # ----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # ----------------------------#
    #   Apply weight smoothing
    # ----------------------------#
    ema = ModelEMA(model_train)

    # ---------------------------#
    #   Load dataset annotations
    # ---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            classes_path=classes_path, model_path=model_path, input_shape=input_shape, \
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
        # ---------------------------------------------------------#
        #   Total training epochs refer to the number of times all data is traversed
        #   Total training steps refer to the total number of gradient descent updates
        #   Each training epoch consists of several steps, with one gradient descent per step.
        #   Below is the minimum recommended number of epochs, with no upper limit.
        # ----------------------------------------------------------#
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('Dataset is too small for training. Please enlarge the dataset.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] For %s optimizer, it is recommended to set total training steps above %d.\033[0m" % (optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] Current total training data size is %d, Unfreeze_batch_size is %d, training for %d epochs, yielding total training steps of %d.\033[0m" % (
                num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] Since total training steps (%d) are less than recommended (%d), consider setting total epochs to %d.\033[0m" % (
                total_step, wanted_step, wanted_epoch))
    # ------------------------------------------------------#
    #   The backbone network's feature extraction is universal.
    #   Freezing training can accelerate training and prevent
    #   backbone weights from being destroyed in early stages.
    #   Init_Epoch is the starting epoch.
    #   Freeze_Epoch is the epoch count for freezing training.
    #   UnFreeze_Epoch is the total epoch count.
    #   Adjust Batch_size if OOM or memory issues occur.
    # ------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        # ------------------------------------#
        #   Freeze certain layers for training
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        # -------------------------------------------------------------------#
        #   If not freezing, set batch_size to Unfreeze_batch_size
        # -------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # -------------------------------------------------------------------#
        #   Adjust learning rate according to the current batch_size
        # -------------------------------------------------------------------#
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # ---------------------------------------#
        #   Choose optimizer based on optimizer_type
        # ---------------------------------------#
        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
            'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        # ---------------------------------------#
        #   Get the learning rate scheduler
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # ---------------------------------------#
        #   Determine the length of each epoch
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training. Please enlarge the dataset.")

        if ema:
            ema.updates = epoch_step * Init_Epoch

        # ---------------------------------------#
        #   Build dataset loaders
        # ---------------------------------------#
        train_dataset = YoloDataset(train_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                    mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob,
                                    train=True, special_aug_ratio=special_aug_ratio)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                  mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False,
                                  special_aug_ratio=0)

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
                         drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        # ----------------------#
        #   Record evaluation mAP curve
        # ----------------------#
        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        # ---------------------------------------#
        #   Start model training
        # ---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # ---------------------------------------#
            #   If there are frozen layers, unfreeze them
            # ---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                # Adjust learning rate based on the new batch_size
                lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue training. Please enlarge the dataset.")

                if ema:
                    ema.updates = epoch_step * epoch

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler,
                                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler,
                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                          epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir,
                          local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
