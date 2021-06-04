from __future__ import print_function
import os, sys
import logging
import argparse
import shutil
import getpass
import subprocess

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class txt_logger:
    def __init__(self, save_folder, opt, argv):
        self.save_folder = save_folder
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        if os.path.isfile(os.path.join(save_folder, 'logfile.log')):
            os.remove(os.path.join(save_folder, 'logfile.log'))

        file_log_handler = logging.FileHandler(os.path.join(save_folder, 'logfile.log'))
        self.logger.addHandler(file_log_handler)

        stdout_log_handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(stdout_log_handler)
        # commend line
        self.logger.info("# COMMEND LINE ===========")
        self.logger.info(argv)
        self.logger.info("# =====================")
        # meta info
        self.logger.info("# META INFO ===========")
        attrs = vars(opt)
        for item in attrs.items():
            self.logger.info("%s: %s"%item)
        # self.logger.info("Saved in: {}".format(save_folder))
        self.logger.info("# =====================")

    def log_value(self, epoch, *info_pairs):
        log_str = "Epoch: {}; ".format(epoch)
        for name, value in info_pairs:
            log_str += (str(name) + ": {}; ").format(value)
        self.logger.info(log_str)

    def save_value(self, name, list_of_values):
        np.save(os.path.join(self.save_folder, name), list_of_values)



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def trim_accuracy(output, target, num_attr_vals):
    '''
    compute the accurcay for every attribute

    return:
        - res: [accuracy for each attribute]
    '''
    with torch.no_grad():
        batch_size = target.size(0)

        pred = (output > 0.5).float()
        correct = pred.eq(target).sum(axis=0).float()/batch_size * 100.0

        res = []
        for attr_val in range(num_attr_vals):
            res.append(correct[attr_val].item())

        return res


class MyReduceLROnPlateau:
    def __init__(self, opt, factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, change_init_lr=True,
                 optimizer=None, stage_info=[(2e+5, 8), (5e+4, -1)], min_lr=1e-5):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        # Attach optimizer
        self.patience = patience
        self.verbose = verbose
        self.lowest_loss = 1e+32
        self.num_bad_epochs = 0
        self.threshold = threshold
        self.opt = opt
        self.change_init_lr = change_init_lr
        self.optimizer = optimizer
        self.num_data_encounter = 0
        self.stage = 0
        self.in_stage_tracker = 0
        self.stage_info = stage_info
        self.call_num_left = stage_info[self.stage][1]
        self.dynamic_call_interval = stage_info[self.stage][0]
        self.last_stage = False
        self.min_lr = min_lr


    def step(self, loss):
        # print("lowest loss: {}; loss now: {}; bad epoch so far: {}".format(self.lowest_loss, loss, self.bad_epoch))
        if loss < self.lowest_loss - self.threshold:
            self.lowest_loss = loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            if self.change_init_lr:
                self.opt.learning_rate = max(self.min_lr, self.opt.learning_rate * self.factor)
                self.num_bad_epochs = 0
                if self.verbose:
                    print("Reduce on Pleateu : init learning rate -> : {}".format(self.opt.learning_rate))

            else:
                self.num_bad_epochs = 0
                for param in self.optimizer.param_groups:
                    lr_ = param['lr']
                    lr = max(self.min_lr, lr_ * self.factor)
                    param['lr'] = lr
                if self.verbose:
                    print("Reduce on Pleateu : learning rate {} -> : {}".format(lr_, lr))

    def batch_step(self, loss, num_data):
        self.num_data_encounter += num_data
        self.change_stage()

        if self.num_data_encounter >= self.dynamic_call_interval:
            if self.opt.lr_scheduling == 'exp_decay':
                lr = self.optimizer.param_groups[0]['lr'] * self.opt.exp_decay_rate
                self.optimizer.param_groups[0]['lr'] = lr

            self.step(loss)
            self.num_data_encounter = 0
            if not self.last_stage:
                self.call_num_left -= 1
            if self.verbose:
                print("Calling scheduler: interval: {}; self.call_num_left: {}".format(self.dynamic_call_interval, self.call_num_left))

    def change_stage(self):
        # decide where to
        if not self.last_stage:
            if self.call_num_left <= 0:
                self.stage += 1
                if self.stage >= len(self.stage_info):
                    print("Enter the last phase")
                    self.stage = -1
                    self.last_stage = True
                self.call_num_left = self.stage_info[self.stage][1]
                self.dynamic_call_interval = self.stage_info[self.stage][0]
                print("Change scheduler stage to ({},{})".format(self.dynamic_call_interval, self.call_num_left))




def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.lr_scheduling == 'adam':
        return None
    elif args.lr_scheduling == 'cosine':
        eta_min = lr * (args.lr_decay_rate ** 3)

        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.lr_scheduling == 'exp_decay':
        if epoch == 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(lr, args.min_lr)
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * args.exp_decay_rate, args.min_lr)

    elif args.lr_scheduling == 'warmup':
        assert args.learning_rate >= args.min_lr, "learning rate should >= min lr"
        warmup_epochs = int(args.epochs * args.warmup_percent)
        up_slope = (args.learning_rate - args.min_lr) / warmup_epochs
        down_slope = (args.learning_rate - args.min_lr) / (args.epochs - warmup_epochs)
        if epoch <= warmup_epochs:
            lr = args.min_lr + up_slope * epoch
        else:
            # lr = args.learning_rate - slope * (epoch - warmup_epochs)
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)

            lr = eta_min + (args.learning_rate - eta_min) * (
                1 + math.cos(math.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs))) / 2

        for param_group in optimizer.param_groups:
            param_group['lr'] = max(lr, args.min_lr)


    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def exclude_bias_and_norm(p):
    return p.ndim == 1



def set_default_path(opt):
    # set the path according to the environment

    if opt.dataset == 'ut-zap50k-sub':
        if not opt.data_folder:
            opt.data_folder = '../data_processing/ut-zap50k-data-subcategory'
        if not opt.data_root_name:
            opt.data_root_name = 'ut-zap50k-images-square'
    elif opt.dataset == 'CUB':
        if not opt.data_folder:
            opt.data_folder = '../data_processing/CUB_200_2011'
        if not opt.data_root_name:
            opt.data_root_name = 'images'
    elif opt.dataset == 'Wider':
        if not opt.data_folder:
            opt.data_folder = '../data_processing/Wider'
        if not opt.data_root_name:
            opt.data_root_name = ''
    elif opt.dataset == 'imagenet100':
        if not opt.data_folder:
            opt.data_folder = '../data_processing/imagenet100'
        if not opt.data_root_name:
            opt.data_root_name = 'imagenet_unzip'
    else:
        raise ValueError(opt.dataset)



def set_optimizer(opt, model, load_opt=True):

    
    optimizer = optim.SGD(model.parameters(),
                        lr=opt.learning_rate,
                        momentum=opt.momentum,
                        weight_decay=opt.weight_decay,
                        nesterov=True)

    
    # load optimizer
    if opt.resume_model_path and load_opt:
        ckpt = torch.load(opt.resume_model_path, map_location='cpu')
        opt_state_dict = ckpt['optimizer']
        opt_new_state_dict = {}
        for k, v in opt_state_dict.items():
            k = k.replace("module.", "")
            opt_new_state_dict[k] = v
        opt_state_dict = opt_new_state_dict
        optimizer.load_state_dict(opt_state_dict)

    elif hasattr(opt, 'resume_linear'):
        # load optimizer for linear
        print("Load optimizer for linear classifier...")
        ckpt = torch.load(os.path.join(
            opt.save_folder, 'classifier_ckpt.pth'), map_location='cpu')
        opt_state_dict = ckpt['optimizer']
        opt_new_state_dict = {}
        for k, v in opt_state_dict.items():
            k = k.replace("module.", "")
            opt_new_state_dict[k] = v
        opt_state_dict = opt_new_state_dict
        optimizer.load_state_dict(opt_state_dict)


    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state



def set_parser(parser_str, additional_argu_func=None, if_linear=False):
    '''
    construct parser and add required argument

    arguments:
        - parser_str: parser name
        - additional_argu_func: function to add more argument
        - if_linear: if linear evaluation
    return:
        a parser object
    '''

    parser = argparse.ArgumentParser(parser_str)

    parser.add_argument('--pipeline', action='store_true', help='decide whether performing pipeline or not, if true, suppressing all std in main')
    parser.add_argument('--save_freq', type=int, default=20, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--save_path', type=str, default='/projects/rsalakhugroup/tianqinl/train_related', help='where to save file')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--method', type=str, default="SupCon")
    parser.add_argument('--instruction', type=str, required=True)
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, required=True, help='what dataset to use')
    parser.add_argument('--data_folder', type=str, default='', help='dataset')
    parser.add_argument('--data_root_name', type=str, default='', help="dataset img folder name, only needed when dataset is organized by folders of img")
    parser.add_argument('--meta_file_train', type=str, default='meta_data_train.csv',
                        help='meta data for ssl training')
    parser.add_argument('--gran_lvl', type=str, required=True, help='what granularity it is using')
    parser.add_argument('--linear_gran_lvl', type=str, default='', help="what granularity the linear will be using")
    parser.add_argument('--img_size', type=int, default=32, required=True, choices=[32, 64, 112, 224], help="image size to train/val")
    parser.add_argument('--customized_name', type=str, default='', help='adding customized name for saving folder')

    parser.add_argument('--overwrite', action='store_true',
                        help='if true, it will allow the program to overwrite any results')
    # seed
    parser.add_argument('--seed', type=int, default=0, help="seed for selecting part of data per class as training")


    # optimization
    optimization_argument(parser)

    # resume training
    parser.add_argument('--resume_model_path', type=str, default='',
                        help='model_path to resume')
    
    if additional_argu_func:
        additional_argu_func(parser)

    opt = parser.parse_args()
    
    set_default_path(opt)

    
    if opt.method == 'SimCLR':
        opt.gran_lvl = '-1'

    

    # get user name
    opt.user_name = getpass.getuser()

    opt.data_root_folder = os.path.join(opt.data_folder, opt.data_root_name)

    return opt

def parser_processing(opt, default_save_folder=True):
    '''
    process parser and create datafolder.

    opt must have model_path, model_name attributes.

    input:
        - opt: argparser object
    '''



    if opt.lr_scheduling == 'cosine':
        opt.change_init_lr = True
    else:
        opt.change_init_lr = False

    if default_save_folder:
        set_save_folder(opt)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder, exist_ok=True)

    opt.tb_folder = os.path.join(opt.save_folder, 'tensorboard')

    if os.path.isdir(opt.tb_folder):
        if not hasattr(opt, 'overwrite'):
            if opt.overwrite == False:
                delete = input("Are you sure to delete folder {}? (Y/n)".format(opt.tb_folder))
        else:
            delete = 'y'
        if delete.lower() == 'y':
            rm_command = "rm -rf " + str(opt.tb_folder)
            os.system(rm_command)
            # shutil.rmtree(opt.tb_folder)
        else:
            sys.exit("{} FOLDER is untouched.".format(opt.tb_folder))

    os.makedirs(opt.tb_folder, exist_ok=True)



    return

def optimization_argument(parser):
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--exp_decay_rate', type=float, default=0.95,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.95,
                        help='momentum')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='SGD min learning rate')
    parser.add_argument('--lr_scheduling', type=str, required=True, choices=['cosine', 'exp_decay', 'adam', 'warmup'], help='what learning rate scheduling to use')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--warmup_percent', type=float, default=0.33,
                        help='percent of epochs that used for warmup')



def set_save_folder(opt):
    opt.model_path = os.path.join(opt.save_path, 'Cl-InfoNCE/{}/{}/train_file_{}/{}_models_granlvl_{}_img_size_{}_{}'.format(opt.dataset, opt.instruction,
                                                                                                                                opt.meta_file_train.replace("meta_data_train_", "").replace(".csv", ""), opt.method,\
                                                                                                                      opt.gran_lvl, opt.img_size, opt.customized_name))
    if not hasattr(opt, 'temp'):
        opt.temp = 'NA'
    opt.model_name = '{}_lr_{}_decay_{}_bsz_{}_temp_{}_scheduling_{}_epochs_{}_trial_{}'.\
                    format(opt.model, opt.learning_rate,
                    opt.weight_decay, opt.batch_size, opt.temp, opt.lr_scheduling, opt.epochs, opt.trial)

    if opt.resume_model_path:
        opt.pre_ssl_epoch = int(opt.resume_model_path.split('/')[-1].split('.')[0].split('_')[-1])
        opt.model_name += '_resume_from_epoch_{}'.format(opt.pre_ssl_epoch)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)




def update_and_save_acc(train_accs_multiple, train_acc, epoch, tf_logger, scalar_logger, key='top1', mode='train'):
    """Update acc result and save it to file"""
    train_accs_multiple[key].append(train_acc[key])
    tf_logger.log_value('Train_Accuaracy ({})'.format(key), train_acc[key], epoch)
    scalar_logger.save_value('{}_acc_{}.npy'.format(mode, key), train_accs_multiple[key])


def suppress_std(func):
    """a decorator that used for suppressing std out when using pipeline"""
    def wrapper(*args, **kwargs):
        stderr_tmp = sys.stderr
        stdout_tmp = sys.stdout
        null = open(os.devnull, 'w')
        sys.stdout = null
        sys.stderr = null
        try:
            result = func(*args, **kwargs)
            sys.stderr = stderr_tmp
            sys.stdout = stdout_tmp
            return result
        except:
            sys.stderr = stderr_tmp
            sys.stdout = stdout_tmp
            raise
    return wrapper


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.00)


