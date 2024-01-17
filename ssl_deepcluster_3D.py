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
from scipy.sparse import csr_matrix

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
    TrainingDataset_SSL
)
from models import ( 
        ResNet,
        ProjectionHead, 
        MultiPrototypes,
        EncoderModel,
        ResNet_Trans
)


logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of 3D-DeepCluster-Universal")



#########################
## changed parameters ###
#########################
parser.add_argument("--log_dir", type=str, default= './log/' , help="path to log file")
parser.add_argument("--tensorboard_name", type=str, default= "universal_deepcluster3D", help="name of tensorboard folder")
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
                    default= "resnet_deepcluster2D.pth", 
                    type=str,
                    help="resnet pretrained path ")
parser.add_argument("--trans_pretrained", 
                    default= "mask_deepcluster.pth",
                    type=str,
                    help="transformer pretrained path ")
parser.add_argument("--checkpoint_name", 
                    default= 'deepcluster3D.pth' , 
                    type=str,
                    help="checkpoint name ")

#########################
#### data parameters ####
#########################
parser.add_argument("--nmb_crops", type=int, default=[1,1], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[128,128], nargs="+",
                    help="crops resolutions (example: [224, 96])")
#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--nmb_prototypes", default=[4,4,4], type=int,
                    help="number of prototypes")
parser.add_argument("--queue_length", type=int, default=3840,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")

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
    n_frames = args.target_shape[0]
    augment_list = [
                    tio.RandomFlip(),
                    tio.RandomAffine(),
                    tio.RandomAnisotropy(),
                    tio.RandomNoise(),
                    tio.RandomBlur(),
                    tio.RandomBiasField(),
                    tio.RandomGamma()
                   ]
    # build data
    all_ds = [ 
            TrainingDataset_SSL(  
                data_path = args.data_path,
                data_name = name_ds,
                augment = augment_list,
                target_shape = args.target_shape
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
    weight = torch.load(args.pretrained_path + args.resnet_pretrained, map_location = 'cpu')
    resnet.load_state_dict(weight, strict = False)
    encoder = EncoderModel(
             n_frames = n_frames, 
             d_model = args.d_model, 
             ffn_hidden = 720,
             drop_prob = 0.1,
             n_layers = 4
    )
    encoder_weight = torch.load(args.pretrained_path + args.trans_pretrained, map_location = 'cpu')["state_dict"]
    remove_prefix = 'module.encoder.'
    encoder_weight = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in encoder_weight.items()}
    encoder.load_state_dict(encoder_weight, strict=False)
    prototypes = MultiPrototypes(output_dim = args.d_model, nmb_prototypes = args.nmb_prototypes)
    model = ResNet_Trans(resnet,encoder,prototypes )
    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
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
    logger.info("Building optimizer done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on]
    )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path,args.checkpoint_name),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
    )
    start_epoch = to_restore["epoch"]

    # build the memory bank
    mb_path = os.path.join(args.dump_path, "mb" + str(args.rank) + ".pth")
    if os.path.isfile(mb_path):
        mb_ckp = torch.load(mb_path)
        local_memory_index = mb_ckp["local_memory_index"]
        local_memory_embeddings = mb_ckp["local_memory_embeddings"]
    else:
        local_memory_index, local_memory_embeddings = init_memory(train_loader, model)

    cudnn.benchmark = True
    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # train the network
        scores, local_memory_index, local_memory_embeddings = train(
            train_loader,
            model,
            optimizer,
            epoch,
            lr_schedule,
            local_memory_index,
            local_memory_embeddings,
        )
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
            
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, args.checkpoint_name),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )
        torch.save({"local_memory_embeddings": local_memory_embeddings,
                    "local_memory_index": local_memory_index}, mb_path)


def train(loader, model, optimizer, epoch, schedule, local_memory_index, local_memory_embeddings):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)

    assignments = cluster_memory(model, local_memory_index, local_memory_embeddings, len(loader.dataset))
    logger.info('Clustering for epoch {} done.'.format(epoch))

    end = time.time()
    start_idx = 0
    for it, data in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = schedule[iteration]

        # ============ multi-res forward passes ... ============
        bs = data['view1'].size(0)
        idx = data['index']
        inputs = torch.cat( [data['view1'], data['view2']],0)
        emb, output = model(inputs)
        emb = emb.detach()
        # ============ deepcluster-v2 loss ... ============
        loss = 0
        for h in range(len(args.nmb_prototypes)):
            scores = output[h] / args.temperature
            targets = assignments[h][idx].repeat(sum(args.nmb_crops)).cuda(non_blocking=True)
            loss += cross_entropy(scores, targets)
        loss /= len(args.nmb_prototypes)

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        loss.backward()
        # cancel some gradients
        if iteration < args.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        optimizer.step()

        # ============ update memory banks ... ============
        local_memory_index[start_idx : start_idx + bs] = data['index']
        for i, crop_idx in enumerate(args.crops_for_assign):
            local_memory_embeddings[i][start_idx : start_idx + bs] = \
                emb[crop_idx * bs : (crop_idx + 1) * bs]
        start_idx += bs

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
    return (epoch, losses.avg), local_memory_index, local_memory_embeddings


def init_memory(dataloader, model):
    size_memory_per_process = len(dataloader) * args.batch_size
    local_memory_index = torch.zeros(size_memory_per_process).long().cuda()
    local_memory_embeddings = torch.zeros(len(args.crops_for_assign), size_memory_per_process, args.d_model).cuda()
    start_idx = 0
    keys = ['view1', 'view2']
    with torch.no_grad():
        logger.info('Start initializing the memory banks')
        for data in dataloader:
            index = data['index']
            nmb_unique_idx = data['view1'].size(0)
            index = index.cuda(non_blocking=True)

            # get embeddings
            outputs = []
            for key in keys:
                inp = data[key].cuda(non_blocking=True)
                outputs.append(model(inp)[0])

            # fill the memory bank
            local_memory_index[start_idx : start_idx + nmb_unique_idx] = index
            for mb_idx, embeddings in enumerate(outputs):
                local_memory_embeddings[mb_idx][
                    start_idx : start_idx + nmb_unique_idx
                ] = embeddings
            start_idx += nmb_unique_idx
    logger.info('Initializion of the memory banks done.')
    return local_memory_index, local_memory_embeddings


def cluster_memory(model, local_memory_index, local_memory_embeddings, size_dataset, nmb_kmeans_iters=10):
    j = 0
    assignments = -100 * torch.ones(len(args.nmb_prototypes), size_dataset).long()
    with torch.no_grad():
        for i_K, K in enumerate(args.nmb_prototypes):
            # run distributed k-means

            # init centroids with elements from memory bank of rank 0
            centroids = torch.empty(K, args.d_model).cuda(non_blocking=True)
            if args.rank == 0:
                random_idx = torch.randperm(len(local_memory_embeddings[j]))[:K]
                assert len(random_idx) >= K, "please reduce the number of centroids"
                centroids = local_memory_embeddings[j][random_idx]

            dist.broadcast(centroids, 0)

            for n_iter in range(nmb_kmeans_iters + 1):

                # E step
                dot_products = torch.mm(local_memory_embeddings[j], centroids.t())
                _, local_assignments = dot_products.max(dim=1)

                # finish
                if n_iter == nmb_kmeans_iters:
                    break

                # M step
                where_helper = get_indices_sparse(local_assignments.cpu().numpy())
                counts = torch.zeros(K).cuda(non_blocking=True).int()
                emb_sums = torch.zeros(K, args.d_model).cuda(non_blocking=True)
                for k in range(len(where_helper)):
                    if len(where_helper[k][0]) > 0:
                        emb_sums[k] = torch.sum(
                            local_memory_embeddings[j][where_helper[k][0]],
                            dim=0,
                        )
                        counts[k] = len(where_helper[k][0])
                dist.all_reduce(counts)
                mask = counts > 0
                dist.all_reduce(emb_sums)
                centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)

                # normalize centroids
                centroids = nn.functional.normalize(centroids, dim=1, p=2)

            getattr(model.module.prototypes, "prototypes" + str(i_K)).weight.copy_(centroids)

            # gather the assignments
            assignments_all = torch.empty(args.world_size, local_assignments.size(0),
                                          dtype=local_assignments.dtype, device=local_assignments.device)
            assignments_all = list(assignments_all.unbind(0))
            dist_process = dist.all_gather(assignments_all, local_assignments, async_op=True)
            dist_process.wait()
            assignments_all = torch.cat(assignments_all).cpu()

            # gather the indexes
            indexes_all = torch.empty(args.world_size, local_memory_index.size(0),
                                      dtype=local_memory_index.dtype, device=local_memory_index.device)
            indexes_all = list(indexes_all.unbind(0))
            dist_process = dist.all_gather(indexes_all, local_memory_index, async_op=True)
            dist_process.wait()
            indexes_all = torch.cat(indexes_all).cpu()

            # log assignments
            assignments[i_K][indexes_all] = assignments_all

            # next memory bank to use
            j = (j + 1) % len(args.crops_for_assign)

    return assignments


def get_indices_sparse(data):
    cols = np.arange(data.size)
    M = csr_matrix((cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size))
    return [np.unravel_index(row.data, data.shape) for row in M]


if __name__ == "__main__":
    main()

