# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
import time
import psutil

import torch
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.utils.data.distributed
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as ct
import settings
import torch.backends.cudnn as cudnn

cudnn.benchmark = False
cudnn.deterministic = True
from settings import logger, formatter
import models
import logging
import utils
import re
#from tensorboardX import SummaryWriter
from datasets import DatasetHDF5
#writer = SummaryWriter()

import models.lstm as lstmpy
from torch.autograd import Variable
import json

from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)
torch.set_num_threads(1)

_support_datasets = ['imagenet', 'cifar10', 'cifar100', 'mnist', 'imdb']
_support_dnns = [
    'resnet50', 'resnet20', 'resnet56', 'resnet110', 'vgg19', 'vgg16',
    'alexnet', 'lstm', 'lstmimdb'
]

#NUM_CPU_THREADS=1
NUM_CPU_THREADS = 0

process = psutil.Process(os.getpid())


def init_processes(rank, size, backend='tcp', master='gpu10'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = master
    os.environ['MASTER_PORT'] = '5935'

    #master_ip = "gpu20"
    #master_mt = '%s://%s:%s' % (backend, master_ip, '5955')
    logger.info("initialized trainer rank: %d of %d......" % (rank, size))
    #dist.init_process_group(backend=backend, init_method=master_mt, rank=rank, world_size=size)
    dist.init_process_group(backend=backend, rank=rank, world_size=size)
    logger.info("finished trainer rank: %d......" % rank)


class wordIndex(object):

    def __init__(self):
        self.count = 0
        self.word_to_idx = {}
        self.word_count = {}

    def add_word(self, word):
        if not word in self.word_to_idx:
            self.word_to_idx[word] = self.count
            self.word_count[word] = 1
            self.count += 1
        else:
            self.word_count[word] += 1

    def add_text(self, text):
        for word in text.split(' '):
            self.add_word(word)


class MnistNet(nn.Module):

    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.name = 'mnistnet'

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class IMDB(Dataset):

    def __init__(self, root, train=True) -> None:
        super().__init__()
        if train:
            f = open(root + '/imdb_train.csv')
        else:
            f = open(root + '/imdb_test.csv')
        self.data = f.readlines()
        self.data.pop(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_available_gpu_device_ids(ngpus):
    return range(0, ngpus)


def create_net(num_classes, dnn='resnet20', **kwargs):
    ext = None
    if dnn in ['resnet20', 'resnet56', 'resnet110']:
        net = models.__dict__[dnn](num_classes=num_classes)
    elif dnn == 'resnet50':
        net = models.__dict__['resnet50'](num_classes=num_classes)
    elif dnn == 'mnistnet':
        net = MnistNet()
    elif dnn in ['vgg16', 'vgg19']:
        net = models.VGG(dnn.upper(), num_classes)
    elif dnn == 'alexnet':
        net = torchvision.models.alexnet()
    elif dnn == 'lstm':
        net = lstmpy.lstm(vocab_size=kwargs['vocab_size'],
                          batch_size=kwargs['batch_size'])
    elif dnn == 'lstmimdb':
        net = models.__dict__['LSTMIMDB'](is_cuda=kwargs['is_cuda'])
    else:
        errstr = 'Unsupport neural network %s' % dnn
        logger.error(errstr)
        raise errstr
    return net, ext


class DLTrainer:

    def __init__(self,
                 rank,
                 size,
                 master='gpu10',
                 dist=True,
                 ngpus=1,
                 batch_size=32,
                 nsteps_update=1,
                 is_weak_scaling=True,
                 data_dir='./data',
                 dataset='cifar10',
                 dnn='resnet20',
                 lr=0.04,
                 nworkers=1,
                 prefix=None,
                 sparsity=0.95,
                 pretrain=None,
                 num_steps=35,
                 tb_writer=None):

        self.size = size
        self.rank = rank
        self.pretrain = pretrain
        self.dataset = dataset
        self.prefix = prefix
        self.num_steps = num_steps
        self.ngpus = ngpus
        self.writer = tb_writer
        self.nsteps_update = nsteps_update
        if self.ngpus > 0:
            self.batch_size = batch_size * self.ngpus if is_weak_scaling else batch_size
        else:
            self.batch_size = batch_size
        self.num_batches_per_epoch = -1
        if self.dataset == 'cifar10' or self.dataset == 'mnist':
            self.num_classes = 10
        elif self.dataset == 'imagenet':
            self.num_classes = 1000
        elif self.dataset == 'cifar100':
            self.num_classes = 100
        elif self.dataset == 'imdb':
            self.num_classes = 2
        self.nworkers = nworkers  # just for easy comparison
        self.data_dir = data_dir
        self.is_cuda = self.ngpus > 0
        if type(dnn) != str:
            self.net = dnn
            self.dnn = dnn.name
            self.ext = None  # leave for further parameters
        else:
            self.dnn = dnn
            # TODO: Refact these codes!
            if self.dnn == 'lstm':
                if data_dir is not None:
                    self.data_prepare()
                self.net, self.ext = create_net(self.num_classes,
                                                self.dnn,
                                                vocab_size=self.vocab_size,
                                                batch_size=self.batch_size)
            elif self.dnn == 'lstmimdb':
                if data_dir is not None:
                    self.data_prepare()
                self.net, self.ext = create_net(self.num_classes,
                                                self.dnn,
                                                is_cuda=self.is_cuda)
            else:
                if data_dir is not None:
                    self.data_prepare()
                self.net, self.ext = create_net(self.num_classes, self.dnn)
        self.lr = lr
        self.base_lr = self.lr
        # if self.is_cuda:
        #     torch.cuda.manual_seed_all(3000)

        if self.is_cuda:
            if self.ngpus > 1:
                devices = get_available_gpu_device_ids(ngpus)
                self.net = torch.nn.DataParallel(self.net,
                                                 device_ids=devices).cuda()
            else:
                self.net.cuda()
        self.net.share_memory()
        self.accuracy = 0
        self.loss = 0.0
        self.train_iter = 0
        self.recved_counter = 0
        self.master = master
        self.average_iter = 0
        if dist:
            init_processes(rank, size, master=master)
        if self.dataset == 'imdb':
            self.criterion = nn.NLLLoss()
        else:
            if self.is_cuda:
                self.criterion = nn.CrossEntropyLoss().cuda()
            else:
                self.criterion = nn.CrossEntropyLoss()
        weight_decay = 1e-4
        self.m = 0.9  # momentum
        nesterov = False
        if self.dataset == 'imagenet':
            weight_decay = 5e-4

        self.optimizer = optim.SGD(self.net.parameters(),
                                   lr=self.lr,
                                   momentum=self.m,
                                   weight_decay=weight_decay,
                                   nesterov=nesterov)

        self.train_epoch = 0

        if self.pretrain is not None and os.path.isfile(self.pretrain):
            self.load_model_from_file(self.pretrain)

        self.sparsities = []
        self.compression_ratios = []
        self.communication_sizes = []
        self.remainer = {}
        self.v = {}  #
        #self.target_sparsities = [0., 0.15, 0.3, 0.6, 0.75, 0.9375, 0.984375, 0.996, 0.999]
        #self.target_sparsities = [0., 0.15, 0.3, 0.6]
        #self.target_sparsities = [0., 0.3, 0.9, 0.95, 0.999]
        #self.target_sparsities = [0., 0.1, 0.15, 0.2, 0.3, 0.5, 0.9, 0.95, 1.]
        self.target_sparsities = [1.]
        self.sparsity = sparsity
        logger.info('target_sparsities: %s', self.target_sparsities)
        self.avg_loss_per_epoch = 0.0
        self.timer = 0.0
        self.iotime = 0.0
        self.epochs_info = []
        self.distributions = {}
        self.gpu_caches = {}
        self.delays = []
        self.num_of_updates_during_comm = 0
        self.train_acc_top1 = []
        logger.info('num_batches_per_epoch: %d' % self.num_batches_per_epoch)

    def get_acc(self):
        return self.accuracy

    def get_loss(self):
        return self.loss

    def get_model_state(self):
        return self.net.state_dict()

    def get_data_shape(self):
        return self._input_shape, self._output_shape

    def get_train_epoch(self):
        return self.train_epoch

    def get_train_iter(self):
        return self.train_iter

    def set_train_epoch(self, epoch):
        self.train_epoch = epoch

    def set_train_iter(self, iteration):
        self.train_iter = iteration

    def load_model_from_file(self, filename):
        checkpoint = torch.load(filename)
        self.net.load_state_dict(checkpoint['state'])
        self.train_epoch = checkpoint['epoch']
        self.train_iter = checkpoint['iter']
        logger.info(
            'Load pretrain model: %s, start from epoch %d and iter: %d',
            filename, self.train_epoch, self.train_iter)

    def get_num_of_training_samples(self):
        return len(self.trainset)

    def imagenet_prepare(self):
        # Data loading code
        traindir = os.path.join(self.data_dir, 'train')
        testdir = os.path.join(self.data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        #trainset = torchvision.datasets.ImageFolder(traindir, transforms.Compose([
        hdf5fn = os.path.join(self.data_dir, 'imagenet-shuffled.hdf5')
        image_size = 224
        self._input_shape = (self.batch_size, 3, image_size, image_size)
        self._output_shape = (self.batch_size, 1000)
        trainset = DatasetHDF5(
            hdf5fn, 'train',
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        self.trainset = trainset

        train_sampler = None
        shuffle = False
        if self.nworkers > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=NUM_CPU_THREADS,
            pin_memory=True,
            sampler=train_sampler)
        #testset = torchvision.datasets.ImageFolder(testdir, transforms.Compose([
        testset = DatasetHDF5(
            hdf5fn,
            'val',
            transforms.Compose([
                transforms.ToPILImage(),
                #        transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        self.testset = testset
        self.testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True)
        #num_workers=1, pin_memory=True)

    def cifar10_prepare(self):
        image_size = 32
        self._input_shape = (self.batch_size, 3, image_size, image_size)
        self._output_shape = (self.batch_size, 10)
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                         std=[0.247, 0.243, 0.262])
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        trainset = torchvision.datasets.CIFAR10(root=self.data_dir,
                                                train=True,
                                                download=True,
                                                transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root=self.data_dir,
                                               train=False,
                                               download=True,
                                               transform=test_transform)
        self.trainset = trainset
        self.testset = testset

        train_sampler = None
        shuffle = True
        if self.nworkers > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler
        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            sampler=train_sampler)
        #shuffle=shuffle, num_workers=NUM_CPU_THREADS, sampler=train_sampler)
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        #shuffle=False, num_workers=1)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
                        'horse', 'ship', 'truck')

    def cifar100_prepare(self):
        image_size = 32
        self._input_shape = (self.batch_size, 3, image_size, image_size)
        self._output_shape = (self.batch_size, 100)
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                         std=[0.247, 0.243, 0.262])
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        trainset = torchvision.datasets.CIFAR100(root=self.data_dir,
                                                 train=True,
                                                 download=True,
                                                 transform=train_transform)
        testset = torchvision.datasets.CIFAR100(root=self.data_dir,
                                                train=False,
                                                download=True,
                                                transform=test_transform)
        self.trainset = trainset
        self.testset = testset

        train_sampler = None
        shuffle = True
        if self.nworkers > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler
        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            sampler=train_sampler)
        #shuffle=shuffle, num_workers=NUM_CPU_THREADS, sampler=train_sampler)
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        #shuffle=False, num_workers=1)
        # self.classes = ('plane', 'car', 'bird', 'cat',
        #        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def mnist_prepare(self):
        image_size = 28
        self._input_shape = (self.batch_size, 3, image_size, image_size)
        self._output_shape = (self.batch_size, 10)
        trainset = torchvision.datasets.MNIST(self.data_dir,
                                              train=True,
                                              download=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(
                                                      (0.1307, ), (0.3081, ))
                                              ]))
        self.trainset = trainset
        testset = torchvision.datasets.MNIST(self.data_dir,
                                             train=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     (0.1307, ), (0.3081, ))
                                             ]))
        self.testset = testset
        train_sampler = None
        shuffle = True
        if self.nworkers > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=NUM_CPU_THREADS,
            sampler=train_sampler)
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        #batch_size=self.batch_size, shuffle=False, num_workers=1)

    def imdb_prepare(self):

        def normalizeString(s):
            s = s.lower().strip()
            s = re.sub(r"<br />", r" ", s)
            # s = re.sub(' +',' ',s)
            s = re.sub(r'(\W)(?=\1)', '', s)
            s = re.sub(r"([.!?])", r" \1", s)
            s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)

            return s

        self.imdb_normalize = normalizeString
        vocabLimit = 50000
        self.imdb_limit = vocabLimit
        self.imdb_max_sequence_len = 500
        obj1 = wordIndex()
        use_cuda = self.is_cuda
        f = open(self.data_dir + '/imdb_train.csv')
        for idx, lines in enumerate(f.readlines()):
            if not idx == 0:
                data = lines.split(',')[1]
                data = normalizeString(data).strip()
                obj1.add_text(data)
        f.close()
        dict1 = sorted(obj1.word_count.items(),
                       key=lambda t: t[1],
                       reverse=True)
        count = 0
        for x, y in dict1:
            if count >= vocabLimit - 1:
                obj1.word_to_idx[x] = vocabLimit
            else:
                obj1.word_to_idx[x] = count
            count += 1
        self.imdb_dict = obj1
        self.trainset = IMDB('./data')
        self.testset = IMDB('./data', False)

        train_sampler = None
        shuffle = True
        if self.nworkers > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler
        assert self.batch_size == 1
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            sampler=train_sampler)
        #shuffle=shuffle, num_workers=NUM_CPU_THREADS, sampler=train_sampler)
        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0)
        #shuffle=False, num_workers=1)

    def data_prepare(self):
        if self.dataset == 'imagenet':
            self.imagenet_prepare()
        elif self.dataset == 'cifar10':
            self.cifar10_prepare()
        elif self.dataset == 'cifar100':
            self.cifar100_prepare()
        elif self.dataset == 'mnist':
            self.mnist_prepare()
        elif self.dataset == 'imdb':
            self.imdb_prepare()
        else:
            errstr = 'Unsupport dataset: %s' % self.dataset
            logger.error(errstr)
            raise errstr
        self.data_iterator = None  #iter(self.trainloader)
        #self.num_batches_per_epoch = (self.get_num_of_training_samples()+self.batch_size*self.nworkers-1)//(self.batch_size*self.nworkers)
        self.num_batches_per_epoch = self.get_num_of_training_samples() // (
            self.batch_size * self.nworkers * self.nsteps_update)

    def update_optimizer(self, optimizer):
        self.optimizer = optimizer

    def update_nworker(self, nworkers, new_rank=-1):
        if new_rank >= 0:
            rank = new_rank
            self.nworkers = nworkers
        else:
            reduced_worker = self.nworkers - nworkers
            rank = self.rank
            if reduced_worker > 0 and self.rank >= reduced_worker:
                rank = self.rank - reduced_worker
        self.rank = rank
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.trainset, num_replicas=nworkers, rank=rank)
        train_sampler.set_epoch(self.train_epoch)
        shuffle = False
        self.train_sampler = train_sampler
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=NUM_CPU_THREADS,
            sampler=train_sampler)
        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1)
        self.nworkers = nworkers
        self.num_batches_per_epoch = (self.get_num_of_training_samples() +
                                      self.batch_size * self.nworkers -
                                      1) // (self.batch_size * self.nworkers)

    def data_iter(self):
        #d = self.data_iterator.next()
        try:
            d = self.data_iterator.next()
        except:
            self.data_iterator = iter(self.trainloader)
            d = self.data_iterator.next()

        if self.dnn != 'lstmimdb' and d[0].size()[0] != self.batch_size:
            return self.data_iter()
        return d

    def _adjust_learning_rate_lstmimdb(self, progress, optimizer):
        first = 10
        second = 20
        third = 30
        if progress < first:
            lr = self.base_lr
        elif progress < second:
            lr = self.base_lr * 0.1
        elif progress < third:
            lr = self.base_lr * 0.01
        else:
            lr = self.base_lr * 0.001
        self.lr = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

    def _adjust_learning_rate_general(self, progress, optimizer):
        warmup = 10
        if settings.WARMUP and progress < warmup:
            warmup_total_iters = self.num_batches_per_epoch * warmup
            min_lr = self.base_lr / self.nworkers
            lr_interval = (self.base_lr - min_lr) / warmup_total_iters
            self.lr = min_lr + lr_interval * self.train_iter
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
            return self.lr
        first = 81
        second = first + 41
        third = second + 33
        if self.dataset == 'imagenet':
            first = 30
            second = 60
            third = 80
        if progress < first:
            lr = self.base_lr
        elif progress < second:
            lr = self.base_lr * 0.1
        elif progress < third:
            lr = self.base_lr * 0.01
        else:
            lr = self.base_lr * 0.001
        self.lr = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

    def adjust_learning_rate(self, progress, optimizer):
        if self.dnn == 'lstmimdb':
            return self._adjust_learning_rate_lstmimdb(progress, optimizer)
        return self._adjust_learning_rate_general(progress, optimizer)

    def print_weight_gradient_ratio(self):
        # Tensorboard
        if self.rank == 0 and self.writer is not None:
            for name, param in self.net.named_parameters():
                self.writer.add_histogram(name,
                                          param.clone().cpu().data.numpy(),
                                          self.train_epoch)
        return

    def finish(self):
        if self.writer is not None:
            self.writer.close()

    def cal_accuracy(self, output, target, topk=(1, )):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(
                    correct_k.mul_(100.0 / batch_size).cpu().data.numpy())
            return res

    def train(self, num_of_iters=1, data=None, hidden=None):
        self.loss = 0.0
        s = time.time()

        for i in range(num_of_iters):
            self.adjust_learning_rate(self.train_epoch, self.optimizer)
            test_time = time.time()
            if self.train_iter % self.num_batches_per_epoch == 0 and self.train_iter > 0:
                logger.info('train iter: %d, num_batches_per_epoch: %d',
                            self.train_iter, self.num_batches_per_epoch)
                #logger.info('Epoch %d, avg train acc: %f, lr: %f, avg loss: %f' % (self.train_iter//self.num_batches_per_epoch, np.mean(self.train_acc_top1), self.lr, self.avg_loss_per_epoch/self.num_batches_per_epoch))
                logger.info(
                    'Epoch %d, avg train acc: %f, lr: %f, avg loss: %f' %
                    (self.train_epoch, np.mean(self.train_acc_top1), self.lr,
                     self.avg_loss_per_epoch / self.num_batches_per_epoch))
                mean_s = np.mean(self.sparsities)
                if self.train_iter > 0 and np.isnan(mean_s):
                    logger.warn('NaN detected! sparsities:  %s' %
                                self.sparsities)
                logger.info(
                    'Average Sparsity: %f, compression ratio: %f, communication size: %f',
                    np.mean(self.sparsities), np.mean(self.compression_ratios),
                    np.mean(self.communication_sizes))
                if self.rank == 0 and self.writer is not None:
                    self.writer.add_scalar(
                        'cross_entropy',
                        self.avg_loss_per_epoch / self.num_batches_per_epoch,
                        self.train_epoch)
                    self.writer.add_scalar('top-1 acc',
                                           np.mean(self.train_acc_top1),
                                           self.train_epoch)
                if self.rank == 0 and (self.train_epoch == 0 or
                                       (self.train_epoch + 1) % 1 == 0):
                    print("test epoch: ", self.train_epoch + 1)
                    self.test(self.train_epoch + 1)
                self.sparsities = []
                self.compression_ratios = []
                self.communication_sizes = []
                self.train_acc_top1 = []
                self.epochs_info.append(self.avg_loss_per_epoch /
                                        self.num_batches_per_epoch)
                self.avg_loss_per_epoch = 0.0
                if self.train_iter > 0 and self.rank == 0:
                    state = {
                        'iter': self.train_iter,
                        'epoch': self.train_epoch,
                        'state': self.get_model_state()
                    }
                    if self.prefix:
                        relative_path = './weights/%s/%s-n%d-bs%d-lr%.4f' % (
                            self.prefix, self.dnn, self.nworkers,
                            self.batch_size, self.base_lr)
                    else:
                        relative_path = './weights/%s-n%d-bs%d-lr%.4f' % (
                            self.dnn, self.nworkers, self.batch_size,
                            self.base_lr)
                    if settings.SPARSE:
                        relative_path += '-s%.5f' % self.sparsity
                    utils.create_path(relative_path)
                    filename = '%s-rank%d-epoch%d.pth' % (self.dnn, self.rank,
                                                          self.train_epoch)
                    fn = os.path.join(relative_path, filename)
                    #self.save_checkpoint(state, fn)
                    #self.remove_dict(state)
                self.train_epoch += 1
                if self.train_sampler and (self.nworkers > 1):
                    print("set train epoch: ", self.train_epoch)
                    self.train_sampler.set_epoch(self.train_epoch)
            this_test_time = time.time() - test_time
            self.the_test_time += this_test_time

            ss = time.time()
            if data is None:
                data = self.data_iter()

            if self.dataset == 'imdb':
                max_sequence_len = self.imdb_max_sequence_len
                labels, data = data[0].split(',')
                labels = int(labels)
                data = self.imdb_normalize(data).strip()
                inputs = [
                    self.imdb_dict.word_to_idx[word]
                    for word in data.split(' ')
                ]
                #print("input data length ", len(inputs))
                if len(inputs) > max_sequence_len:
                    inputs = inputs[0:max_sequence_len]
            else:
                inputs, labels_cpu = data
            if self.is_cuda:
                if self.dnn == 'lstm':
                    inputs = Variable(inputs.transpose(0,
                                                       1).contiguous()).cuda()
                    labels = Variable(labels_cpu.transpose(
                        0, 1).contiguous()).cuda()
                elif self.dnn == 'lstmimdb':
                    inputs = Variable(torch.cuda.LongTensor(inputs))
                    labels = Variable(torch.cuda.LongTensor([labels]))
                else:
                    inputs, labels = inputs.cuda(
                        non_blocking=True), labels_cpu.cuda(non_blocking=True)
            else:
                labels = labels_cpu

            self.iotime += (time.time() - ss)
            com_time = time.time()

            if self.dnn == 'lstm':
                hidden = lstmpy.repackage_hidden(hidden)
                outputs, hidden = self.net(inputs, hidden)
                tt = torch.squeeze(
                    labels.view(-1, self.net.batch_size * self.net.num_steps))
                loss = self.criterion(outputs.view(-1, self.net.vocab_size),
                                      tt)
                loss.backward()
            elif self.dnn == 'lstmimdb':
                hidden = self.net.init_hidden()
                y_pred, _ = self.net(inputs, hidden)
                self.net.zero_grad()
                loss = self.criterion(y_pred, labels)
                loss.backward()
            else:
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
            self.comp_time = time.time() - com_time
            loss_value = loss.item()
            # logger.info statistics
            self.loss += loss_value

            self.avg_loss_per_epoch += loss_value

            if self.dnn not in ['lstm', 'lstmimdb']:
                acc1, = self.cal_accuracy(outputs, labels, topk=(1, ))
                self.train_acc_top1.append(acc1)

            self.train_iter += 1
        self.num_of_updates_during_comm += 1
        self.loss /= num_of_iters
        self.timer += time.time() - s
        #display = 100
        display = self.num_batches_per_epoch
        if self.train_iter % display == 0 and self.rank == 0:
            logger.info(
                '[%3d][%5d/%5d][rank:%d] loss: %.3f, average forward and backward time: %f, iotime: %f '
                % (self.train_epoch, self.train_iter,
                   self.num_batches_per_epoch, self.rank, self.loss,
                   self.timer / display, self.iotime / display))
            mbytes = 1024. * 1024
            logger.info(
                'GPU memory usage memory_allocated: %d MBytes, max_memory_allocated: %d MBytes, memory_cached: %d MBytes, max_memory_cached: %d MBytes, CPU memory usage: %d MBytes',
                ct.memory_allocated() / mbytes,
                ct.max_memory_allocated() / mbytes,
                ct.memory_cached() / mbytes,
                ct.max_memory_cached() / mbytes,
                process.memory_info().rss / mbytes)
            self.timer = 0.0
            self.iotime = 0.0
            if self.is_cuda:
                torch.cuda.empty_cache()

        if self.dnn == 'lstm':
            return num_of_iters, hidden
        return num_of_iters

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        total_steps = 0
        costs = 0.0
        total_iters = 0
        total_wer = 0
        # logger.info('test')
        for batch_idx, data in enumerate(self.testloader):
            # logger.info(batch_idx)
            if self.dataset == 'imdb':
                lines = data[0]
                data = self.imdb_normalize(lines.split(',')[1]).strip()
                input_data = []
                vocabLimit = self.imdb_limit
                for word in data.split(' '):
                    if not word in self.imdb_dict.word_to_idx:
                        input_data.append(vocabLimit)
                    else:
                        input_data.append(self.imdb_dict.word_to_idx[word])
                if len(input_data) > self.imdb_max_sequence_len:
                    input_data = input_data[0:self.imdb_max_sequence_len]

            else:
                inputs, labels_cpu = data
            if self.is_cuda:
                if self.dnn == 'lstm':
                    inputs = Variable(inputs.transpose(0,
                                                       1).contiguous()).cuda()
                    labels = Variable(labels_cpu.transpose(
                        0, 1).contiguous()).cuda()
                if self.dnn == 'lstmimdb':
                    input_data = Variable(torch.cuda.LongTensor(input_data))
                else:
                    inputs, labels = inputs.cuda(
                        non_blocking=True), labels_cpu.cuda(non_blocking=True)
            else:
                labels = labels_cpu

            if self.dnn == 'lstm':
                hidden = self.net.init_hidden()
                hidden = lstmpy.repackage_hidden(hidden)
                outputs, hidden = self.net(inputs, hidden)
                tt = torch.squeeze(
                    labels.view(-1, self.net.batch_size * self.net.num_steps))
                loss = self.criterion(outputs.view(-1, self.net.vocab_size),
                                      tt)
                test_loss += loss.data[0]
                costs += loss.data[0] * self.net.num_steps
                total_steps += self.net.num_steps
            elif self.dnn == 'lstmimdb':

                hidden = self.net.init_hidden()
                y_pred, _ = self.net(input_data, hidden)
                pred1 = y_pred.max(1)[1].cpu().numpy()
                if pred1 == int(lines.split(',')[0]):
                    correct += 1
            else:
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.data.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(labels.data).cpu().sum()
            if self.dnn == 'lstmimdb':
                total += 1
            else:
                total += labels.size(0)
            total_iters += 1
        test_loss /= total_iters
        if self.dnn not in ['lstm']:
            acc = float(correct) / total
        elif self.dnn == 'lstm':
            acc = np.exp(costs / total_steps)

        loss = float(test_loss) / total
        logger.info('Epoch %d, lr: %f, val loss: %f, val acc: %f' %
                    (epoch, self.lr, test_loss, acc))
        self.net.train()
        return acc

    def update_model(self):
        self.optimizer.step()

    def remove_dict(self, dictionary):
        dictionary.clear()

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)

    def zero_grad(self):
        self.optimizer.zero_grad()


def train_with_single(dnn,
                      dataset,
                      data_dir,
                      nworkers,
                      lr,
                      batch_size,
                      nsteps_update,
                      max_epochs,
                      num_steps=1):
    torch.cuda.set_device(0)
    trainer = DLTrainer(0,
                        nworkers,
                        dist=False,
                        batch_size=batch_size,
                        is_weak_scaling=True,
                        ngpus=1,
                        data_dir=data_dir,
                        dataset=dataset,
                        dnn=dnn,
                        lr=lr,
                        nworkers=nworkers,
                        prefix='singlegpu',
                        num_steps=num_steps)
    trainer.the_test_time = 0
    iters_per_epoch = trainer.get_num_of_training_samples() // (
        nworkers * batch_size * nsteps_update)

    times = []
    display = 100 if iters_per_epoch > 100 else iters_per_epoch - 1
    for epoch in range(max_epochs):
        if dnn == 'lstm':
            hidden = trainer.net.init_hidden()
        for i in range(iters_per_epoch):
            s = time.time()
            trainer.optimizer.zero_grad()
            for j in range(nsteps_update):
                if dnn == 'lstm':
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)
            trainer.update_model()
            times.append(time.time() - s)
            if i % display == 0 and i > 0:
                time_per_iter = np.mean(times)
                logger.info(
                    'Time per iteration including communication: %f. Speed: %f images/s',
                    time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single trainer")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--nsteps-update', type=int, default=1)
    parser.add_argument('--dataset',
                        type=str,
                        default='imagenet',
                        choices=_support_datasets,
                        help='Specify the dataset for training')
    parser.add_argument('--dnn',
                        type=str,
                        default='resnet50',
                        choices=_support_dnns,
                        help='Specify the neural network for training')
    parser.add_argument('--data-dir',
                        type=str,
                        default='./data',
                        help='Specify the data root path')
    parser.add_argument('--lr',
                        type=float,
                        default=0.1,
                        help='Default learning rate')
    parser.add_argument('--max-epochs',
                        type=int,
                        default=90,
                        help='Default maximum epochs to train')
    parser.add_argument('--num-steps', type=int, default=35)
    args = parser.parse_args()
    batch_size = args.batch_size * args.nsteps_update
    prefix = settings.PREFIX
    relative_path = './logs/singlegpu-%s/%s-n%d-bs%d-lr%.4f-ns%d' % (
        prefix, args.dnn, 1, batch_size, args.lr, args.nsteps_update)
    utils.create_path(relative_path)
    logfile = os.path.join(relative_path, settings.hostname + '.log')
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.info('Configurations: %s', args)
    train_with_single(args.dnn, args.dataset, args.data_dir, 1, args.lr,
                      args.batch_size, args.nsteps_update, args.max_epochs,
                      args.num_steps)
