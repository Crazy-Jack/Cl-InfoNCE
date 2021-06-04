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
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from tqdm import tqdm
from torchsummary import summary
import numpy as np

from data_utils import DynamicLabelDataset
from data_utils import MyTransform
from util import AverageMeter
from util import adjust_learning_rate
from util import set_optimizer, save_model
from util import txt_logger, set_parser, parser_processing, suppress_std, set_save_folder
from networks.resnet_big import SupConResNet, SupCEResNet
from losses import SupConLoss

try:
    import apex
    from apex import amp, optimizers
except:
    print("Can't import apex...")
    pass


def additional_argument(parser):
    parser.add_argument('--used_attributes', type=int, nargs='+', default=[-1], help='Most informative attributes')
    parser.add_argument('--print_freq', type=int, default=100, help='print freq')

    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    return


def parse_option():
    parser_str = 'argument for training'
    opt = set_parser(parser_str, additional_argu_func=additional_argument, if_linear=False)
    if opt.method == 'WeakSupCon':
        if opt.used_attributes == [-1]:
            opt.used_attributes = [i for i in range(opt.gran_lvl)]
        else:
            assert len(opt.used_attributes) == opt.gran_lvl

    parser_processing(opt, default_save_folder=True)

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

    
    if opt.dataset in ['ut-zap50k-sub', 'CUB', 'Wider', 'imagenet100']:
        train_meta_df = pd.read_csv(os.path.join(opt.data_folder, opt.instruction, opt.meta_file_train), index_col=0)
        
        train_dataset = DynamicLabelDataset(df=train_meta_df, data_path=opt.data_root_folder,
                                gran_lvl=opt.gran_lvl, transform = train_transform)
            
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    opt.n_cls = train_dataset.num_class
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=False, sampler=train_sampler)

    return train_loader


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


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter('BT')
    data_time = AverageMeter('DT')
    losses = AverageMeter('Loss')

    end = time.time()
    # for idx, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
    for idx, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data_time.update(time.time() - end)
        if opt.method in ['SupCon', 'SimCLR']:
            images = torch.cat([images[0], images[1]], dim=0)
        images = images.cuda()
        labels = labels.cuda()
        bsz = labels.shape[0]

        # get features
        features = model(images)
        
        ## handle augmentation
        if opt.method in ['SupCon', 'SimCLR']:
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) # [bz, 2, feature_dim]
        
        ## handle loss
        if opt.method in ['SupCon', 'CE']:
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                            format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if ((idx + 1) % opt.print_freq == 0) or (idx == 0):
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

        
    return losses.avg


def main(opt):

    # tensorboard and logger
    tf_logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    scalar_logger = txt_logger(opt.save_folder, opt, 'python ' + ' '.join(sys.argv))


    # build data loader
    train_loader = set_loader(opt)

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

    for epoch in range(start + 1, opt.epochs + 1):
        # adjust lr
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        start_time_epoch = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
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



if __name__ == '__main__':
    
    # get options
    opt = parse_option()
    save_folder = main(opt)
    print("ckpt#{}".format(save_folder))
