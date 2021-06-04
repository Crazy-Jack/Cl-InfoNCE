from __future__ import print_function

import sys
import os
import argparse
import time
import math
import shutil
import re

import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torchvision import transforms, datasets
import pandas as pd
import numpy as np

from data_utils import DynamicLabelDataset
from data_utils import MyTransform
from util import AverageMeter
from util import txt_logger, parser_processing, optimization_argument
from util import adjust_learning_rate, accuracy
from util import set_optimizer, save_model
from util import update_and_save_acc
from main_clinfonce import parse_option
from networks.resnet_big import SupConResNet, LinearClassifier, SupCEResNet
import tensorboard_logger as tb_logger


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

def parsing_logfile(opt):
    '''
    read previous logfile located in ckpt directory and load them into opt
    '''
    model_dir = os.path.dirname(opt.ckpt)
    flag = False
    opt_dict = {}
    # read logfile
    with open(os.path.join(model_dir, 'logfile.log'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "META INFO" in line and not flag:
                # start META INFO session
                flag = True
            elif flag:
                if "=====================" in line:
                    # end META INFO session
                    flag = False
                    break
                else:
                    line = line.strip('\n')
                    (name, val) = line.split(': ')
                    opt_dict[name] = val

    # type conversion
    if "resume_model_path" in opt_dict:
        opt_dict["resume_model_path"] = ''
    for name, val in opt_dict.items():
        if re.fullmatch(r'\d+', val) or val == '-1':
            opt_dict[name] = int(val)
        elif re.fullmatch(r'[\d\.\-e]+', val):
            opt_dict[name] = float(val)
        elif val == 'True':
            opt_dict[name] = True
        elif val == 'False':
            opt_dict[name] = False


        if not hasattr(opt, name):
            setattr(opt, name, opt_dict[name])

    return opt_dict


def set_parser():
    '''
    return parser opt
    '''
    parser = argparse.ArgumentParser('Argument for Linear Evaluation')
    parser.add_argument('--linear_gran_lvl', type=str, default='0', help='meta data for linear training/testing')
    parser.add_argument('--ckpt', type=str, required=True, help='path to pre-trained model')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--meta_file_train', type=str, default='meta_data_train.csv',
                        help='meta data for ssl training')
    parser.add_argument('--meta_file_test', type=str, default='meta_data_val.csv',
                        help='meta data for ssl testing')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--instruction', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true',
                        help='if true, it will allow the program to overwrite any results')
    parser.add_argument('--load_memory', action='store_true',
                        help='if true, it will load all the data into memory to avoid bottleneck of I/O')
    parser.add_argument('--method', type=str, required=True, help='which model to load')

    # training details
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    


    # optimization
    optimization_argument(parser)
    opt = parser.parse_args()

    if hasattr(opt, 'stage_info'):
        opt.stage_info = [(int(i.split(":")[0]), int(i.split(":")[1])) for i in opt.stage_info.split(",")]

    return opt


def parse_option():
    opt = set_parser()
    original_opt_dict = parsing_logfile(opt)
    opt.ssl_epoch = int(opt.ckpt.split('/')[-1].split('.')[0].split('_')[-1])
    opt.save_folder = os.path.join(opt.save_folder, 'linear_eval_epoch_{}'.format(opt.ssl_epoch))
    parser_processing(opt, default_save_folder=False)
    return opt

def set_loader(opt):
    mytransform = MyTransform(opt)
    train_transform = mytransform.train_transform(ssl=False)
    val_transform = mytransform.val_transform()


    if opt.dataset in ['imagenet100', 'full_imagenet']:
        train_meta_df = pd.read_csv(os.path.join(opt.data_folder, opt.instruction, opt.meta_file_train), index_col=0)
        val_meta_df = pd.read_csv(os.path.join(opt.data_folder, opt.instruction, opt.meta_file_test), index_col=0)
        train_dataset = DynamicLabelDataset(df=train_meta_df, data_path=opt.data_root_folder,
                                            gran_lvl = opt.linear_gran_lvl, transform = train_transform)
        val_dataset = DynamicLabelDataset(df=val_meta_df, data_path=opt.data_folder,
                                        gran_lvl = opt.linear_gran_lvl, transform = val_transform)
    elif opt.dataset in ['ut-zap50k', 'ut-zap50k-sub', 'CUB', 'Wider']:
        train_meta_df = pd.read_csv(os.path.join(opt.data_folder, opt.instruction, opt.meta_file_train), index_col=0)
        val_meta_df = pd.read_csv(os.path.join(opt.data_folder, opt.instruction, opt.meta_file_test), index_col=0)
        
        train_dataset = DynamicLabelDataset(df=train_meta_df, data_path=opt.data_root_folder,
                                        gran_lvl = opt.linear_gran_lvl, transform = train_transform)
        val_dataset = DynamicLabelDataset(df=val_meta_df, data_path=opt.data_root_folder,
                                        gran_lvl = opt.linear_gran_lvl, transform = val_transform)
    else:
        raise ValueError(opt.dataset)

    opt.n_cls = train_dataset.num_class
    print("inside main linear, n_cls", opt.n_cls)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=False, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=False)

    return train_loader, val_loader




def set_model(opt):
    # get weight and meta info
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']
    start_epoch = 1
    try:
        previous_num_class = int(ckpt['opt'].n_cls)
    except:
        if opt.method == 'CE':
            previous_num_class = ckpt['model']['fc.bias']
        else:
            previous_num_class = 0

    if opt.method in ['SupCon', 'SimCLR']:
        model = SupConResNet(name=opt.model)
    elif opt.method == 'CE':
        model = SupCEResNet(name=opt.model, num_classes=previous_num_class, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    # load classifier
    if opt.resume:
        load_classifier_file = os.path.join(
            opt.save_folder, 'classifier_ckpt.pth')
        classifier_ckpt = torch.load(load_classifier_file)
        classifier_state_dict = classifier_ckpt['model']
        new_classifier_state_dict = {}
        for k, v in classifier_state_dict.items():
            k = k.replace(".module.module", ".module")
            new_classifier_state_dict[k] = v
        classifier.load_state_dict(new_classifier_state_dict)
        print("Classifier weight loaded!")
        # load epoch
        start_epoch = classifier_ckpt['epoch']

    # load encoder model
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print("Used devices: {}".format(torch.cuda.device_count()))
            model.encoder = torch.nn.DataParallel(model.encoder)

        new_state_dict = {}
        has_module = False
        for i in model.state_dict().keys():
            if 'module' in i:
                has_module = True
        for k, v in state_dict.items():
            if k[:11] == 'module.head':
                k = k.replace("module.head", "head")

            new_state_dict[k] = v
        state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion, start_epoch


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter('BT')
    data_time = AverageMeter('DT')
    losses = AverageMeter('Loss')
    top1 = AverageMeter('top1')
    topk = [1,]
    final_acc = {'top1': 0}
    if opt.n_cls >= 3:
        top3 = AverageMeter('top3')
        topk.append(3)
        final_acc['top3'] = 0
    if opt.n_cls >= 5:
        top5 = AverageMeter('top5')
        topk.append(5)
        final_acc['top5'] = 0
    if opt.n_cls >= 10:
        top10 = AverageMeter('top10')
        topk.append(10)
        final_acc['top10'] = 0

    end = time.time()
    for idx, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        
        acc = accuracy(output, labels, topk=topk)
        top1.update(acc[0], bsz)
        if opt.n_cls >= 3:
            top3.update(acc[1], bsz)
        if opt.n_cls >= 5:
            top5.update(acc[2], bsz)
        if opt.n_cls >= 10:
            top10.update(acc[3], bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'train acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'train acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'train acc@10 {top10.val:.3f} ({top10.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, top10=top10))
            sys.stdout.flush()
        del images, features

    final_acc['top1'] = top1.avg
    if opt.n_cls >= 3:
        final_acc['top3'] = top3.avg
    if opt.n_cls >= 5:
        final_acc['top5'] = top5.avg
    if opt.n_cls >= 10:
        final_acc['top10'] = top10.avg

    return losses.avg, final_acc


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter('bt')
    losses = AverageMeter('loss')
    top1 = AverageMeter('top1')
    topk = [1,]
    final_acc = {'top1': 0}
    if opt.n_cls >= 3:
        top3 = AverageMeter('top3')
        topk.append(3)
        final_acc['top3'] = 0
    if opt.n_cls >= 5:
        top5 = AverageMeter('top5')
        topk.append(5)
        final_acc['top5'] = 0
    if opt.n_cls >= 10:
        top10 = AverageMeter('top10')
        topk.append(10)
        final_acc['top10'] = 0

    end = time.time()
    with torch.no_grad():
        for idx, (images, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
            
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)

            acc = accuracy(output, labels, topk=topk)
            top1.update(acc[0], bsz)
            if opt.n_cls >= 3:
                top3.update(acc[1], bsz)
            if opt.n_cls >= 5:
                top5.update(acc[2], bsz)
            if opt.n_cls >= 10:
                top10.update(acc[3], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if ((idx + 1) % opt.print_freq == 0) or (idx == 0):
                print('Val: [{0}/{1}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'val acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'val acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                    'val acc@10 {top10.val:.3f} ({top10.avg:.3f})\t'.format(
                    idx + 1, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1, top5=top5, top10=top10))
                sys.stdout.flush()

    final_acc['top1'] = top1.avg
    if opt.n_cls >= 3:
        final_acc['top3'] = top3.avg
    if opt.n_cls >= 5:
        final_acc['top5'] = top5.avg
    if opt.n_cls >= 10:
        final_acc['top10'] = top10.avg


    return losses.avg, final_acc



def main():
    best_acc = 0
    train_accs_multiple = {'top1':[]}
    val_accs_multiple = {'top1':[]}
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion, start_epoch = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # tensorboard and logger
    tf_logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    scalar_logger = txt_logger(opt.tb_folder, opt, 'python ' + ' '.join(sys.argv))

    if opt.n_cls >= 3:
        train_accs_multiple['top3'] = []
        val_accs_multiple['top3'] = []
    if opt.n_cls >= 5:
        train_accs_multiple['top5'] = []
        val_accs_multiple['top5'] = []
    if opt.n_cls >= 10:
        train_accs_multiple['top10'] = []
        val_accs_multiple['top10'] = []

    # training routine
    # determine the start epoch
    for epoch in range(start_epoch, opt.epochs + 1):
        start_time = time.time()
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        loss, train_acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        update_and_save_acc(train_accs_multiple, train_acc, epoch, tf_logger, scalar_logger, key='top1', mode='train')
        if opt.n_cls >= 3:
            update_and_save_acc(train_accs_multiple, train_acc, epoch, tf_logger, scalar_logger, key='top3', mode='train')
        if opt.n_cls >= 5:
            update_and_save_acc(train_accs_multiple, train_acc, epoch, tf_logger, scalar_logger, key='top5', mode='train')
        if opt.n_cls >= 10:
            update_and_save_acc(train_accs_multiple, train_acc, epoch, tf_logger, scalar_logger, key='top10', mode='train')

        tf_logger.log_value('Train_loss', loss, epoch)

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        update_and_save_acc(val_accs_multiple, val_acc, epoch, tf_logger, scalar_logger, key='top1', mode='val')
        if opt.n_cls >= 3:
            update_and_save_acc(val_accs_multiple, val_acc, epoch, tf_logger, scalar_logger, key='top3', mode='val')
        if opt.n_cls >= 5:
            update_and_save_acc(val_accs_multiple, val_acc, epoch, tf_logger, scalar_logger, key='top5', mode='val')
        if opt.n_cls >= 10:
            update_and_save_acc(val_accs_multiple, val_acc, epoch, tf_logger, scalar_logger, key='top10', mode='val')

        tf_logger.log_value('Val_loss', loss, epoch)

        end_time = time.time()
        # log critical info to files
        scalar_logger.log_value(epoch, ('train acc', train_acc),
                                       ('val acc', val_acc),
                                       ('lr', optimizer.param_groups[0]['lr']),
                                       ('time: ', "{} s".format(end_time - start_time)))
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'classifier_ckpt.pth'.format(epoch=epoch))
            save_model(classifier, optimizer, opt, epoch, save_file)

    # save the last model
    if epoch % opt.save_freq != 0:
        save_file = os.path.join(
            opt.save_folder, 'classifier_ckpt.pth'.format(epoch=epoch))
        save_model(classifier, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
