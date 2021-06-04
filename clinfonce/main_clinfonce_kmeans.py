# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import argparse
import time
import math
import pandas as pd
import shutil
import multiprocessing

import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from tqdm import tqdm
from torchsummary import summary
import numpy as np
import faiss

from data_utils import DynamicLabelDatasetIndex
from data_utils import MyTransform
from util import AverageMeter, ProgressMeter
from util import adjust_learning_rate
from util import set_optimizer, save_model
from util import txt_logger, set_parser, parser_processing, suppress_std, set_save_folder
from networks.resnet_big import SupConResNet, SupCEResNet
from losses import SupConLoss


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

def additional_argument(parser):
    parser.add_argument('--used_attributes', type=int, nargs='+', default=[-1], help='Most informative attributes')
    parser.add_argument('--print_freq', type=int, default=100, help='print freq')

    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--warmup_epoch', type=int, default=100)
    parser.add_argument('--perform_cluster_epoch', type=int, default=1)
    parser.add_argument('--low_dim', type=int, default=128, help="latent dim")
    parser.add_argument('--num_cluster', type=str, default="2500", help="num of cluster")


    return


def parse_option():
    parser_str = 'argument for training'
    opt = set_parser(parser_str, additional_argu_func=additional_argument, if_linear=False)

    parser_processing(opt, default_save_folder=True)

    if "," in opt.num_cluster:
        opt.num_cluster = [int(i) for i in opt.num_cluster.split(',')]
    else:
        opt.num_cluster = [int(opt.num_cluster)]
    return opt


def set_loader(opt):
    # costomize transform
    mytransform = MyTransform(opt)
    if opt.method in ['SupCon', 'SimCLR']:
        train_transform = mytransform.train_transform(ssl=True)
    elif opt.method == 'CE':
        train_transform = mytransform.train_transform(ssl=False)
    else:
        raise NotImplementedError("Method {} is not supported for transform operation.".format(opt.method))

    val_transform = mytransform.val_transform()

    # contruct dataset
    if opt.dataset in ['ut-zap50k', 'ut-zap50k-sub', 'CUB', 'aPascal', 'Wider', 'Google_Image', 'imagenet100', 'full_imagenet']:
        train_meta_df = pd.read_csv(os.path.join(opt.data_folder, opt.instruction, opt.meta_file_train), index_col=0)

        train_dataset = DynamicLabelDatasetIndex(df=train_meta_df, data_path=opt.data_root_folder,
                                gran_lvl=opt.gran_lvl, transform = train_transform)
        train_dataset2 = DynamicLabelDatasetIndex(df=train_meta_df, data_path=opt.data_root_folder,
                                gran_lvl=opt.gran_lvl, transform = val_transform)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    opt.n_cls = train_dataset.num_class
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=False, sampler=train_sampler)

    train_loader2 = torch.utils.data.DataLoader(
        train_dataset2, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=False, sampler=train_sampler)
    return train_loader, train_loader2


def set_model(opt, logger):
    if opt.method in ['SupCon', 'SimCLR']:
        model = SupConResNet(name=opt.model)
    elif opt.method == 'CE':
        model = SupCEResNet(name=opt.model, num_classes=opt.n_cls, verbose=True)

    
    if opt.method == 'SupCon' or opt.method == 'SimCLR':
        criterion = SupConLoss(temperature=opt.temp)
    elif opt.method == 'CE':
        print("Setting creterion to Cross_entropy")
        criterion = torch.nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)


    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print("Used devices: {}".format(torch.cuda.device_count()))
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        if opt.resume_model_path:
            # get pre ssl epoch
            ckpt = torch.load(opt.resume_model_path, map_location='cpu')
            state_dict = ckpt['model']
            new_state_dict = {}
            for k, v in state_dict.items():
                if torch.cuda.device_count() > 1:
                    pass
                    # print(k)
                    #if k.split(".")[0] != 'head':
                    #    k = ".".join([k.split(".")[0], "module"] + k.split(".")[1:])
                else:
                    k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
            model.load_state_dict(state_dict)

            logger.logger.info("Model loaded! Pretrained from epoch {}".format(opt.pre_ssl_epoch))

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt, cluster_result=None):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter('BT')
    data_time = AverageMeter('DT')
    losses = AverageMeter('Loss', ':6.2f')
    display = [batch_time, data_time, losses]

    progress = ProgressMeter(
        len(train_loader),
        display,
        prefix="Epoch: [{}]".format(epoch))


    end = time.time()

    for idx, (images, labels, index) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data_time.update(time.time() - end)
        if opt.method in ['SupCon', 'SimCLR']:
            images = torch.cat([images[0], images[1]], dim=0)
        images = images.cuda()
        labels = labels.cuda()
        bsz = labels.shape[0]

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) # [bz, 2, feature_dim]

        if cluster_result is None:
            if opt.method in ['SupCon', 'CE']:
                loss = criterion(features, labels)
            elif opt.method == 'SimCLR':
                loss = criterion(features)
            else:
                raise ValueError('contrastive method not supported: {}'.
                                format(opt.method))
        else:
            # switch to clustering labels
            assert len(cluster_result['im2cluster']) == 1, f"cluster_result['im2cluster'] has {len(cluster_result['im2cluster'])} items, but 1 is required."
            cluster_i = cluster_result['im2cluster'][0]
            cluster_labels = cluster_i[index]
            loss = criterion(features, cluster_labels) # no actual weak label involved
            
        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % opt.print_freq == 0:
            progress.display(idx)

    return losses.avg


# ============== clustering ==================
def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()
    # free GPU for that batch
    torch.cuda.empty_cache()
    features = torch.zeros(len(eval_loader.dataset),args.low_dim).cuda()
    for i, (images, labels, index) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            feat = model(images)
            features[index] = feat
    return features.cpu()


def perform_clustering(args, epoch, eval_loader, model):

    if epoch>=args.warmup_epoch:
        if (epoch+1) % args.perform_cluster_epoch == 0:
            # compute momentum features for center-cropped images
            features = compute_features(eval_loader, model, args)

            # placeholder for clustering result
            cluster_result = {'im2cluster':[],'centroids':[],'density':[]}
            for num_cluster in args.num_cluster:
                cluster_result['im2cluster'].append(torch.zeros(len(eval_loader.dataset),dtype=torch.long).cuda())
                cluster_result['centroids'].append(torch.zeros(int(num_cluster),args.low_dim).cuda())
                cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())


            features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice
            features = features.numpy()
            cluster_result = run_kmeans(features,args)  #run kmeans clustering on master node
            # save the clustering result
            if (epoch+1) % args.save_freq == 0:
                print("\nSaving cluster results...\n")
                torch.save(cluster_result, os.path.join(args.save_folder, 'clusters_%d'%epoch))

            return cluster_result


def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """

    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}

    for seed, num_cluster in enumerate(args.num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 0
        index = faiss.GpuIndexFlatL2(res, d, cfg)

        clus.train(x, index)

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]

        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)

        # sample-to-centroid distances for each cluster
        Dcluster = [[] for c in range(k)]
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

        # concentration estimation (phi)
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)
                density[i] = d

        #if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        density = args.temp*density/density.mean()  #scale the mean to temperature

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)

        im2cluster = torch.LongTensor(im2cluster).cuda()
        density = torch.Tensor(density).cuda()

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)

    return results





def main(opt):

    # tensorboard and logger
    tf_logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    scalar_logger = txt_logger(opt.save_folder, opt, 'python ' + ' '.join(sys.argv))


    # build data loader
    train_loader,eval_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt, scalar_logger)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # training routine
    # resume model path
    if opt.resume_model_path:
        start = opt.pre_ssl_epoch
    else:
        start = 0

    cluster_result = None
    for epoch in range(start + 1, opt.epochs + 1):
        # clustering
        c_result = perform_clustering(opt, epoch, eval_loader, model)
        if c_result:
            cluster_result = c_result

        # adjust lr
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        start_time_epoch = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt, cluster_result=cluster_result)
        end_time_epoch = time.time()
        
        # tensorboard logger
        tf_logger.log_value('loss', loss, epoch)
        tf_logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # file logger
        scalar_logger.log_value(epoch, ('loss', loss),
                                    ('learning_rate', optimizer.param_groups[0]['lr']),
                                    ('time: ', "{} s".format(end_time_epoch - start_time_epoch)))


        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    if epoch % opt.save_freq != 0:
        save_file = os.path.join(
            opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        save_model(model, optimizer, opt, opt.epochs, save_file)

    return opt.save_folder










@suppress_std
def suppress_std_main(opt):
    save_folder = main(opt)
    return save_folder



if __name__ == '__main__':
    """
    Command:
    - Super Con:
        - utzap50k:
        $ python main_supcon.py --batch_size 128 --utzap_gran_lvl 3 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 10
        - deepfashion
        $ python main_supcon.py --batch_size 128 --deepfashion_gran_lvl 0 --dataset deepfashion --data_root_name img --learning_rate 0.5 --temp 0.1 --cosine --data_folder /home/tianqinl/Code/WeakSupervisionSSL/data_unzip/deepfashion
        - CUB (attribute loss)
        $ python main_supcon.py --batch_size 128 --dataset CUB --data_root_name images --data_folder ../data_unzip/CUB_200_2011 --method WeakSupCon --learning_rate 0.5 --temp 0.1
        - CUB (original resnet50)
        $ python main_supcon.py --batch_size 20 --dataset CUB --data_root_name images --data_folder ../data_unzip/CUB_200_2011 --method WeakSupCon --learning_rate 0.5 --temp 0.1 --img_size 224 --model resnet50_original
    - SimCLR:
        - utzap50k:
        $ python main_supcon.py --batch_size 128 --method SimCLR --learning_rate 0.5 --temp 0.5 --cosine
        - deepfashion
        $ python main_supcon.py --batch_size 128 --method SimCLR --dataset deepfashion --data_root_name img --learning_rate 0.5 --temp 0.5 --cosine --data_folder /home/tianqinl/Code/WeakSupervisionSSL/data_unzip/deepfashion

    - Adding loss of weak label
        - utzap50k:
        $ python main_supcon.py --meta_file_train meta_data_lossadd_train.csv --method WeakSupCon --learning_rate 0.5 --temp 0.1 --cosine --batch_size 128 --used_attributes 0 1 2 3 4 5 6 7 8 9 --utzap_gran_lvl 10 --data_folder /projects/rsalakhugroup/tianqinl/ut-zap50K-processed


    """
    # get options
    opt = parse_option()

    # decide suppress std or not
    if opt.pipeline:
        save_folder = suppress_std_main(opt)
        #save_folder = main(opt)
        sys.stdout.write(save_folder + "\n")
    else:
        save_folder = main(opt)
        print("ckpt#{}".format(save_folder))
