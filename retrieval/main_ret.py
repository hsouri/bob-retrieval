#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
import yaml
import io

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.data import Dataset
# import models.moco_vits as moco_vits
import utils
from utils import extract_features
import pickle
from PIL import Image, ImageFile
import natsort

import timm

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import models


class SynthDataset(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        all_imgs = natsort.natsorted(all_imgs)
        self.total_imgs = [x for x in all_imgs if x.endswith(('.JPG', '.JPEG', '.jpg','.png','.PNG'))]

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image,idx



parser = argparse.ArgumentParser('Generic image retrieval given a path')
parser.add_argument('--data_path', type=str)
parser.add_argument('--dataset', default='', type=str) 
parser.add_argument('--multiscale', default=False, type=utils.bool_flag)
parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
parser.add_argument('--pt_style', default='dino', type=str)

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')

parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


### mae configs
parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')


parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')
parser.add_argument('--global_pool', default=None, type=str)
# parser.set_defaults(global_pool=True)
parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                    help='Use class token instead of global pool for classification')
# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--similarity_metric', default='dotproduct', type=str)
parser.add_argument('--num_loss_chunks', default=1, type=int)
parser.add_argument('--numpatches', default=1, type=int)
parser.add_argument('--layer', default=1, type=int, help="layer from end to create descriptors from.")
parser.add_argument('--stype', default='', type=str,choices=['cross'])
parser.add_argument('--keephead', action='store_true')
parser.add_argument('--keeppredictor', action='store_true')

parser.add_argument('-ssp','--sim_save_path', type=str,default='./similarityscores/')
parser.add_argument('--extra', default='', type=str)
parser.add_argument('--einsum_chunks', default=30, type=int)
parser.add_argument('--dontsave', action='store_true')

parser.add_argument('--num_matches', default=4, type=int)

## for oxfrd paris
parser.add_argument('--imsize', default=224, type=int, help='Image size')

parser.add_argument('--noeval', action='store_true')

best_acc1 = 0


def main():
    args = parser.parse_args()


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    # utils.init_distributed_mode(args)
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # create model
    
        
    if args.pt_style == 'timm': # timm.list_models(pretrained=True)
        model = timm.create_model(args.arch,num_classes=0,pretrained=True)

    elif args.pt_style == 'moco':
        if args.arch == 'resnet50':
            from models import resnet50_moco
            model = resnet50_moco(pretrained=True)
        elif args.arch == 'vit_small_patch16_224':
            from models import vit_small_moco
            model = vit_small_moco(pretrained=True)
        elif args.arch == 'vit_base_patch16_224':
            from models import vit_base_moco
            model = vit_base_moco(pretrained=True)

    elif args.pt_style == 'clip':
        if args.arch == 'vit_base_patch16_224':
            from models import vit_base_patch16_224_clip_laion2b
            model = vit_base_patch16_224_clip_laion2b(pretrained=True)
        elif args.arch == "vit_large_patch14_224":
            from models import vit_large_patch14_224_clip_laion2b
            model = vit_large_patch14_224_clip_laion2b(pretrained=True)
        elif args.arch == "resnet50":
            from models import resnet50_clip
            model = resnet50_clip(pretrained=True)
        elif args.arch == "resnet101":
            from models import resnet101_clip
            model = resnet101_clip(pretrained=True)
        elif args.arch == "resnet50_64":
            from models import resnet50_64_clip
            model = resnet50_64_clip(pretrained=True)
        else:
            raise Exception("This model doesnt exist")
        
    elif args.pt_style == 'mae':
        if args.arch == 'vit_base_patch16_224':
            from models import vit_base_patch16_224_mae
            model = vit_base_patch16_224_mae(pretrained=True)
        elif args.arch == 'vit_large_patch16_224':
            from models import vit_large_patch16_224_mae
            model = vit_large_patch16_224_mae(pretrained=True)

    elif args.pt_style == 'vicreg':
        if args.arch == 'resnet50':
            from models import resnet50_vicreg
            model = resnet50_vicreg(pretrained=True)
    elif args.pt_style == 'midas':
        if args.arch == 'swin2_tiny_256':
            from models import swin2_tiny_256_midas
            model = swin2_tiny_256_midas(pretrained=True)
        elif args.arch == 'swin2_base_384':
            from models import swin2_base_384_midas
            model = swin2_base_384_midas(pretrained=True)
        else:
            NotImplementedError('This model type does not exist for this pt style')
   
   

    if '384' in args.arch:
        args.batch_size = 32

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    if "384" in args.arch:
            transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize(400, interpolation=3),
                transforms.CenterCrop(384),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    elif "resnet50_64" in args.arch:
            transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize(460, interpolation=3),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    elif "256" in args.arch or args.pt_style == 'stablediffusion':
            transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    elif "swinv2_tiny" in args.arch:
            transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize(224, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        #elif 'vit' in args.arch or 'swin' in args.arch or args.pt_style in ['clip','vicregl','sscd','multigrain']:
    else:
            transform = transforms.Compose([
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    # Data loading code
    if args.dataset in ['roxford5k', 'rparis6k']:
        args.data_path = './datasets/revisitop/data/datasets/'

        from utils import OxfordParisDataset
        dataset_values = OxfordParisDataset(args.data_path, args.dataset, split="train", transform=transform, imsize=args.imsize)
        dataset_query = OxfordParisDataset(args.data_path, args.dataset, split="query", transform=transform, imsize=args.imsize)
    elif args.dataset == 'INSTRE': 
        args.data_path = './datasets/moco-v3/data/INSTRE-S1'
        from utils import TwoLeveldataset
        dataset_query = TwoLeveldataset(args.data_path,'query' ,transform, 50)
        dataset_values = TwoLeveldataset(args.data_path,'value' ,transform, 50)

    elif args.dataset == 'CUB200':
        
        args.data_path = './datasets/moco-v3/data/CUB_200_2011/images'
        from utils import TwoLeveldataset
        dataset_query = TwoLeveldataset(args.data_path,'query' ,transform)
        dataset_values = TwoLeveldataset(args.data_path,'value' ,transform)

    elif args.dataset in ['Copydays']:

        args.data_path = './datasets/moco-v3/data/copydays'
        from utils import Copydaysdataset
        dataset_query = Copydaysdataset(args.data_path,'query' ,transform)
        dataset_values = Copydaysdataset(args.data_path,'value' ,transform)
    elif args.dataset == 'Objectnet':
        args.data_path = './datasets/ObjectNet/objectnet-1.0/images'
        from utils import TwoLeveldataset
        dataset_query = TwoLeveldataset(args.data_path,'query' ,transform,50)
        dataset_values = TwoLeveldataset(args.data_path,'value' ,transform,50)
    elif args.dataset == 'iNat':
        args.data_path = "./datasets/ret_datasets/iNat_validation"
        from utils import TwoLeveldataset
        dataset_query = TwoLeveldataset(args.data_path,'query' ,transform, 5)
        dataset_values = TwoLeveldataset(args.data_path,'value' ,transform, 5)

    elif args.dataset == 'google_landmark':
        args.data_path = './datasets/ret_datasets/GoogleLandmarksRet/'
        from utils import GoogleLMRet
        dataset_query = GoogleLMRet(args.data_path, 'query' ,transform)
        dataset_values =  GoogleLMRet(args.data_path, 'value' ,transform)
    else:
    
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        ret_transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ])

        query_dir = os.path.join(args.data_path, 'query')
        val_dir = os.path.join(args.data_path, 'values')
        dataset_query = SynthDataset(query_dir, ret_transform)
        dataset_values = SynthDataset(val_dir, ret_transform)

    ## creating dataloader
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset_values, shuffle=False)
    else:
        sampler = None
    data_loader_values = torch.utils.data.DataLoader(
        dataset_values,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_query = torch.utils.data.DataLoader(
        dataset_query,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"train: {len(dataset_values)} imgs / query: {len(dataset_query)} imgs")

    model.eval()
    ############################################################################
    if not args.multiprocessing_distributed:
        utils.init_distributed_mode(args)
    if args.rank == 0:  # only rank 0 will work from now on
        if args.data_path is not None and 'synthetic' in args.data_path:
            ast = args.data_path
            args.dataset = '_'.join(ast.split('/')[2:])
        mod = args.pretrained
        mod_name = mod.split("/")[-1].split(".")[0]
        if args.keephead:
           mod_name+= '_whd'
        if args.keeppredictor:
             mod_name+= '_wpre'
             
        
        # normalize features
        
        # Step 1: extract features
        values_features = extract_features(args,model, data_loader_values, args.gpu, multiscale=args.multiscale)
        query_features = extract_features(args,model, data_loader_query, args.gpu, multiscale=args.multiscale)
        # if args.similarity_metric == 'dotproduct':
        values_features = nn.functional.normalize(values_features, dim=1, p=2)
        query_features = nn.functional.normalize(query_features, dim=1, p=2)
        # else:
        # import ipdb; ipdb.set_trace()
        if not args.noeval: 
            import wandb
            wandb.init(project="bb_retrieval",name=f"{args.pt_style}-{args.arch}")
            wandb.config.update(args)

        ############################################################################
        # Step 2: similarity
        # if args.similarity_metric == 'splitloss':
            

        #     if v.shape[0] > 10000 or q.shape[0] > 10000:
        #         sim = einsum_in_chunks(v,q,args.stype,args.einsum_chunks)
        #         if 'ddpmeval' in args.dataset or 'celeba' in args.dataset or 'guidediff_imagenet' in args.dataset :
        #             sim2 = einsum_in_chunks(v,v,args.stype,args.einsum_chunks)
        #     else:
        #         chunk_dp = torch.einsum('ncp,mcp->nmc', [v, q])
        #         sim = reduce(chunk_dp, 'n m c -> n m', 'max')
        #         if 'ddpmeval' in args.dataset or 'celeba' in args.dataset or 'guidediff_imagenet' in args.dataset :
        #             chunk_dp2 = torch.einsum('ncp,mcp->nmc', [v, v])
        #             sim2 = reduce(chunk_dp2, 'n m c -> n m', 'max')
        #     # which locations are used in max loss
        #     # chunk_dp_rearr = rearrange(chunk_dp,'n m c -> (n m) c')
        #     # locs = torch.argmax(chunk_dp_rearr, dim=1)
        #     # locations =  locs.cpu().numpy()
        # else:
            # values_features = nn.functional.normalize(values_features, dim=-1, p=2)
            # query_features = nn.functional.normalize(query_features, dim=-1, p=2)
        sim = torch.mm(values_features, query_features.T)



        ranks = torch.argsort(-sim, dim=0).cpu().numpy()
        if not args.dontsave:        
            savepath = f'{args.sim_save_path}/{args.pt_style}_{args.arch}_{mod_name}/{args.similarity_metric}/{args.dataset}{args.extra}/{args.layer}'

            os.makedirs(savepath,exist_ok=True)
            torch.save(sim.cpu(), os.path.join(savepath, "similarity.pth"))

        ############################################################################
        if not args.noeval:
            ks = [1, 5, 10]
            
            # Step 3: evaluate
            if args.dataset in ['roxford5k', 'rparis6k']:
                gnd = dataset_values.cfg['gnd']

                # search for easy & hard
                gnd_t = []
                for i in range(len(gnd)):
                    g = {}
                    g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
                    g['junk'] = np.concatenate([gnd[i]['junk']])
                    gnd_t.append(g)
                map,  pr , recalls, mrr = utils.compute_map(ranks, gnd_t, ks)
            elif args.dataset in ['INSTRE','CUB200','Objectnet', 'iNat','google_landmark']:
                newcfg = dataset_query.newcfg

                map,  pr , recalls, mrr = utils.compute_map(ranks, newcfg, ks)
                
            elif args.dataset in ['Copydays']:
                newcfg = dataset_values.qvmapping
                map,  pr , recalls, mrr = utils.compute_map(ranks, newcfg, ks)
                # from utils import micro_average_precision
                # microAP =  micro_average_precision(newcfg,sim.T)
            

            else:
                with open(f"{args.data_path}/qvmapping.p", 'rb') as f:
                    newcfg = pickle.load(f)
                map,  pr , recalls, mrr = utils.compute_map(ranks, newcfg, ks)
                
            print('>> mAP {:.2f}'.format(np.around(map*100, decimals=2)))

            
            
            wandb.log({
                    'map': map*100 ,
                    'prs' : pr,
                    'recalls' : recalls,
                    'MRR' : mrr,
                    # 'microAP' : microAP
                    })
    dist.barrier()
    
def einsum_in_chunks(v,q,stype='cross',nchunks=100):
    from einops import rearrange, reduce
    n = v.shape[0]
    sim_list = []
    tchunks = torch.chunk(v,nchunks, dim=0)
    count = 0
    for val in tchunks:
        print(f'In chunk {count}')
        # val = v[i,:,:]
        # val = torch.unsqueeze(val, 0)
        if stype=='cross':
            chunk_dp = torch.einsum('ncp,mdp->nmcd', [val,q])
            sim = reduce(chunk_dp, 'n m c d -> n m', 'max').clone()
        else:
            chunk_dp = torch.einsum('ncp,mcp->nmc', [val,q])
            sim = reduce(chunk_dp, 'n m c -> n m', 'max').clone()
        sim_list.append(sim)
        count+=1

    return torch.cat(sim_list,dim=0)

if __name__ == '__main__':
    main()
