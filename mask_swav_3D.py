# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys 
import math
import time
import random
import shutil
import argparse
import numpy as np
from logging import getLogger


import torch
import torchio as tio   # 3d medical image lib 
import torch.optim
import torch.nn as nn
import torchio as tio 
import torch.nn.parallel
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import ConcatDataset

import apex
from apex.parallel.LARC import LARC

sys.path.append('./utils/')

from utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)
from dataset import (
    TrainingDataset_Mask
)
from models import ( 
        ResNet,
        EncoderDecoderModel
)



logger = getLogger()
parser = argparse.ArgumentParser(description="Implementation of MASK-SwAV-universal")

#########################
## changed parameters ###
#########################
parser.add_argument("--log_dir", type=str, default= './log/' , help="path to log file")
parser.add_argument("--tensorboard_name", type=str, default= "universal_mask_swav", help="name of tensorboard folder")
parser.add_argument("--data_path", default= './training_data_3D/', 
                    type=str,
                    help="path to dataset folder")
parser.add_argument("--target_shape", default=(64,128,128), type=tuple,
                    help="target shape for each input data")
parser.add_argument("--pretrained_path", 
                    default= './weights/', 
                    type=str,
                    help="path to pretrained weights")
parser.add_argument("--resnet_pretrained", 
                    default= "resnet_swav2D.pth", 
                    type=str,
                    help="resnet pretrained name ")
parser.add_argument("--checkpoint_name", 
                    default= 'mask_swav.pth' , 
                    type=str,
                    help="checkpoint name ")
#########################
#### data parameters ####
#########################
parser.add_argument("--mask_ratio", type=int, default=0.1, help="mask hidden ratio")
#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=4, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=1e-2, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")
parser.add_argument("--d_model", default= 640, type=int,help="output dim")
#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=False,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")



def main():
    global args
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")
    writer = SummaryWriter(log_dir = args.log_dir + args.tensorboard_name)
    
    augment_list = [
                        tio.RandomFlip(),
                        tio.RandomAffine(),
                        tio.RandomAnisotropy(),
                        tio.RandomNoise(),
                        tio.RandomBlur(),
                        tio.RandomBiasField(),
                        tio.RandomGamma()
                   ]

    n_frames = args.target_shape[0]
    all_ds = [ 
                TrainingDataset_Mask(  
                                    data_path = args.data_path,
                                    data_name = name_ds,
                                    augment = augment_list,
                                    target_shape = args.target_shape,
                                    mask_ratio = args.mask_ratio
                                    )
                        for name_ds in os.listdir(args.data_path)
             ]
    concat_dataset = ConcatDataset(all_ds)
    sampler = torch.utils.data.distributed.DistributedSampler(concat_dataset)
    train_loader = DataLoader(
        concat_dataset, 
        batch_size= args.batch_size,  
        sampler=sampler,
        drop_last = True,
        num_workers = args.workers,
        pin_memory=True
    )    
    logger.info("Building data done with {} images loaded and {} length train loader.".format(len(concat_dataset), len(train_loader)))

    # build model
    resnet = ResNet(normalize = False)
    weight = torch.load(args.pretrained_path + args.resnet_pretrained, map_location='cpu')
    resnet.load_state_dict(weight, strict = False)
    model = EncoderDecoderModel(
             n_frames = n_frames, 
             d_model = args.d_model, 
             ffn_hidden = 720,
             drop_prob = 0.1,
             mask_ratio = args.mask_ratio, 
             n_layers = 4
    )
    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        resnet = nn.SyncBatchNorm.convert_sync_batchnorm(resnet)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        resnet = apex.parallel.convert_syncbn_model(resnet, process_group=process_group)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
        
    # copy model to GPU
    resnet = resnet.cuda()
    model = model.cuda()
    if args.rank == 0:
        logger.info(resnet)
        logger.info(model)
    logger.info("Building model done.")
    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )

    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    
    if args.use_fp16:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        logger.info("Initializing mixed precision done.")

    # wrap model
    resnet = nn.parallel.DistributedDataParallel(
        resnet,
        device_ids=[args.gpu_to_work_on]
    )
    
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on]
    )
    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, args.checkpoint_name),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        amp=apex.amp,
    )
    start_epoch = to_restore["epoch"]

    cudnn.benchmark = True
          
    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # train the network
        scores = train(train_loader,resnet, model, optimizer, epoch, lr_schedule)
        training_stats.update(scores)

        writer.add_scalars('loss', {'train': scores[1] }, epoch)
        writer.add_scalars('lr',   {'train': optimizer.optim.param_groups[0]["lr"] }, epoch)
        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if args.use_fp16 :
                save_dict["amp"] = apex.amp.state_dict()
            logger.info(f"$$$$$$$$$$$$  SAVE MODEL AT EPOCH {epoch} $$$$$$$$$$$$")
            torch.save(
                save_dict,
                os.path.join(args.dump_path, args.checkpoint_name),
            )
            shutil.copyfile(
                    os.path.join(args.dump_path, args.checkpoint_name),
                    os.path.join(args.pretrained_path, args.checkpoint_name),
                )
            
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, args.checkpoint_name),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )
    
    
    
def train(train_loader, resnet, model, optimizer, epoch, lr_schedule):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    resnet.eval()
    model.train()
    end = time.time()
    
    criterion = nn.MSELoss()
    n_frames = args.target_shape[0]
    n_mask_tokens = int(n_frames * args.mask_ratio )
    
    for it, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # ============ multi-res forward passes ... ============
#         bs = data['view1'].shape[0]
#         output = resnet(data['view1'].float())
#         # loss
#         loss = criterion(output,  model(output, data['mask'])  ) 
#         loss /= (n_mask_tokens)

        loss = 0
        bs = data['view1'].shape[0]
        concat_data = torch.cat( [data['view1'], data['view2']] ,0)
        x = resnet(concat_data.float())
        loss1 = criterion(x[:bs],  model(x[:bs], data['mask'])  ) 
        loss2 = criterion(x[bs:],  model(x[bs:], data['mask'])  )
        loss = loss1 + loss2
        loss /= (n_mask_tokens)
        # zero the parameter gradients
        optimizer.zero_grad()
        if args.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
            #loss.backward(retain_graph = True)
        # forward + backward + optimize
        optimizer.step()
        del concat_data
        # ============ misc ... ============
        losses.update(loss.item(), bs)
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.optim.param_groups[0]["lr"],
                )
            )
    return (epoch, losses.avg)
    
    
    
if __name__ == "__main__":
    main()

