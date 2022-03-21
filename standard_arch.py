import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as func
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchmodels
import models.resnet
import models.vgg


models_dict = torchmodels.__dict__
models_dict.update({f"{name}-custom": model for name, model in models.resnet.__dict__.items() if name.startswith("resnet")})
models_dict.update({f"{name}-custom": model for name, model in models.vgg.__dict__.items() if name.startswith("vgg")})
model_names = sorted(name for name in models_dict
                     if name.islower() and not name.startswith("__")
                     and callable(models_dict[name]))


class NonLin(nn.Module):
    depth = 0  # Global counter for when the non-linearity depends on the depth.

    def __init__(self, num_channels, non_linearity_name, init_gain=1.0, init_bias=0.0):
        super().__init__()

        if isinstance(non_linearity_name, str):
            self.non_linearity = non_linearity_name
        elif isinstance(non_linearity_name, list):
            self.non_linearity = non_linearity_name[self.depth % len(non_linearity_name)]
            self.__class__.depth += 1
        else:
            assert False

        self.gain = nn.Parameter(torch.full((num_channels,), fill_value=init_gain), requires_grad=self.non_linearity in ["sigmoid", "pow"])
        self.bias = nn.Parameter(torch.full((num_channels,), fill_value=init_bias), requires_grad=self.non_linearity in ["absdead, thresh", "pow"])

    def extra_repr(self) -> str:
        return f"num_channels={self.bias.shape[0]}, non_linearity={self.non_linearity}"

    def forward(self, x):
        if self.non_linearity == "relu":
            return func.relu(x)
        elif self.non_linearity == "abs":
            return torch.abs(x)
        elif self.non_linearity == "absdead":
            bias = self.bias[(slice(None),) + (None,) * (x.ndim - 2)]
            return func.relu(x - bias) + func.relu(-x - bias)
        elif self.non_linearity == "thresh":
            bias = self.bias[(slice(None),) + (None,) * (x.ndim - 2)]
            return func.relu(x - bias) - func.relu(-x - bias)
        elif self.non_linearity == "tanh":
            return func.tanh(x)
        elif self.non_linearity == "sigmoid":
            return func.sigmoid(self.gain[:, None, None] * x.abs() + self.bias[:, None, None]) / (x.abs() + 1e-6) * x
        elif self.non_linearity == "pow":
            t = self.bias[:, None, None].exp() * x.abs() ** self.gain[:, None, None].exp()
            return t / ((1 + t) * (x.abs() + 1e-6)) * x
        elif self.non_linearity == "powfixed":
            return x / (1 + x.abs())
        else:
            assert False


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dir', default='default_dir', type=str,
                    help='directory for training logs, checkpoints, PCA, whitening')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--loose-resume', action='store_true',
                    help='perform a non-strict resume (ignores missing/unexpected keys and shape errors)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
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
parser.add_argument('--no-bias', action='store_true', help='remove bias from the architectures')
parser.add_argument('--non-linearity', default="relu", nargs="+", help="non-linearity to use")
parser.add_argument('--init-gain', default=1.0, type=float, help="initialization of gain parameter of non-linearities")
parser.add_argument('--init-bias', default=0.0, type=float, help="initialization of bias parameter of non-linearities")

best_acc1 = 0
best_acc5 = 0


logfiles = []


def new_logfile(path):
    """ Returns the logfile, which is an open file (needs to be closed). """
    logfile = open(path, 'a')
    logfiles.append(logfile)
    return logfile


def print_and_write(log, *logfiles):
    """ Prints log (a string) and writes it to logfiles, which should have a write method (open files) or None. """
    log = str(log)
    print(log)
    for logfile in logfiles:
        if logfile is not None:
            logfile.write(log + '\n')


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
    global best_acc1, best_acc5

    file_suffix = f"batchsize_{args.batch_size}"

    checkpoint_savedir = os.path.join('./checkpoints', args.dir)
    if not os.path.exists(checkpoint_savedir):
        os.makedirs(checkpoint_savedir)
    checkpoint_savefile = os.path.join(checkpoint_savedir, f'{file_suffix}.pth.tar')
    best_checkpoint_savefile = os.path.join(checkpoint_savedir, f'{file_suffix}_best.pth.tar')

    logs_dir = os.path.join('./training_logs', args.dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logfile = new_logfile(os.path.join(logs_dir, f'{file_suffix}.log'))
    summaryfile = new_logfile(os.path.join(logs_dir, 'summary_file.txt'))
    writer = SummaryWriter(logs_dir)

    args.gpu = gpu
    if args.gpu is not None:
        print_and_write("Use GPU: {} for training".format(args.gpu), logfile, summaryfile)

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print_and_write("=> using pre-trained model '{}'".format(args.arch), logfile, summaryfile)
        model = models_dict[args.arch](pretrained=True)
    else:
        print_and_write(f"=> creating model {args.arch} with non-linearity {args.non_linearity} and bias {not args.no_bias}", logfile, summaryfile)
        model = models_dict[args.arch](
            non_linearity=NonLin, non_linearity_name=args.non_linearity, bias=not args.no_bias,
            init_gain=args.init_gain, init_bias=args.init_bias,
        )
        print_and_write(model, logfile, summaryfile)

    if not torch.cuda.is_available():
        print_and_write('using CPU, this will be slow', logfile, summaryfile)
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
            args.batch_size = int(args.batch_size / ngpus_per_node)
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

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_and_write("=> loading checkpoint '{}'".format(args.resume), logfile, summaryfile)
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)

            try:
                model.module.load_state_dict(checkpoint['state_dict'])
            except RuntimeError as err:
                if args.loose_resume:
                    print_and_write(f"Loose resume, ignored errors: {err}", logfile, summaryfile)
                else:
                    raise err
            # Do not load the optimizer in loose resume mode, as the parameters will not match.
            if not args.loose_resume:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print_and_write("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']), logfile, summaryfile)
        else:
            print_and_write("=> no checkpoint found at '{}'".format(args.resume), logfile, summaryfile)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, 0, args, writer, logfile, summaryfile)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer, logfile, summaryfile)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, epoch, args, writer, logfile, summaryfile)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_epoch_acc1 = epoch
        if acc5 > best_acc5:
            best_acc5 = acc5
            best_epoch_acc5 = epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint_filename=checkpoint_savefile, best_checkpoint_filename=best_checkpoint_savefile)


def train(train_loader, model, criterion, optimizer, epoch, args, writer, logfile, summaryfile):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch), logfile=logfile)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    if writer is not None:
        suffix = "train"
        writer.add_scalar(f'top5_{suffix}', top5.avg, global_step=epoch)
        writer.add_scalar(f'top1_{suffix}', top1.avg, global_step=epoch)


def validate(val_loader, model, criterion, epoch, args, writer, logfile, summaryfile):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ', logfile=logfile)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print_and_write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5), logfile)

    if writer is not None:
        suffix = "val"
        writer.add_scalar(f'top5_{suffix}', top5.avg, global_step=epoch)
        writer.add_scalar(f'top1_{suffix}', top1.avg, global_step=epoch)

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, checkpoint_filename='checkpoint.pth.tar',
                    best_checkpoint_filename='model_best.pth.tar'):
    torch.save(state, checkpoint_filename)
    if is_best:
        shutil.copyfile(checkpoint_filename, best_checkpoint_filename)


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
    def __init__(self, num_batches, meters, prefix, logfile):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logfile = logfile

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_and_write('\t'.join(entries), self.logfile)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
