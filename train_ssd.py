# from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

from data import *
# from data.voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT
# from data.voc07_consistency_init import  VOCDetection_con_init, VOCAnnotationTransform_con_init, VOC_CLASSES, VOC_ROOT
# from data.voc07_consistency import  VOCDetection_con, VOCAnnotationTransform_con, VOC_CLASSES, VOC_ROOT
# from data.coco import COCODetection, COCOAnnotationTransform, COCO_CLASSES, COCO_ROOT, get_label_map
# from data.config import *

from eval_new import get_output_dir, evaluate_detections
from tqdm import tqdm

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC300', choices=['VOC300', 'VOC512', 'VOC0712_300'],
                    type=str, help='VOC300 or VOC512')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--warmup_period', default=30000, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--save_iter', default=1000, type=int,
                    help='Save weights at this iteration')
parser.add_argument('--save_epoch', default=5, type=int,
                    help='Save weights at this iteration')
parser.add_argument('--num_workers', default=1, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--model_name', default='ssd',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.visdom:
    import visdom
    viz = visdom.Visdom()

def train():
    print(f"args.dataset={args.dataset}")
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        run_val = False
        train_dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC300':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc300
        run_val = True
        from data import VOC_CLASSES as labelmap
        train_dataset = VOCDetection(root=args.dataset_root,
                                     image_sets=[('2007', 'train')],
                                     transform=SSDAugmentation(cfg['min_dim'],
                                                                MEANS))
        
        val_dataset = VOCDetection(root=args.dataset_root,
                                    image_sets=[('2007', 'val')],
                                    transform=SSDAugmentation(cfg['min_dim'],
                                                                MEANS))
    elif args.dataset == 'VOC0712_300':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc300
        run_val = False
        from data import VOC_CLASSES as labelmap
        print("RAN VOC0712!!!!")
        train_dataset = VOCDetection(root=args.dataset_root,
                                     image_sets=[('2007', 'train')],
                                     transform=SSDAugmentation(cfg['min_dim'],
                                                                VOC0712_MEANS,
                                                                VOC0712_STD))
        # for debugging dataset
        # for idx, (data1, image) in enumerate(train_dataset): print(idx, data1, image)

        
        # val_dataset = VOCDetection(root=args.dataset_root,
        #                             image_sets=[('2007', 'val')],
        #                             transform=SSDAugmentation(cfg['min_dim'],
        #                                                         MEANS))
    elif args.dataset == 'VOC512':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc512
        run_val = False
        from data import VOC_CLASSES as labelmap
        train_dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True
    else:
        net=ssd_net

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(train_dataset) // args.batch_size
    print(f'Training SSD on: {train_dataset.name}, epoch size: {epoch_size} iters')
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + train_dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    train_data_loader = data.DataLoader(train_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  generator=torch.Generator(device='cuda'),
                                  pin_memory=True)
    if run_val:
        val_data_loader = data.DataLoader(val_dataset, args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False, collate_fn=detection_collate,
                                    generator=torch.Generator(device='cuda'),
                                    pin_memory=True)

    import pytorch_warmup as warmup 
    # lr_scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    # sched_list = []
    # for step in cfg['lr_steps']:
    #     lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=args.gamma)
    #     sched_list.append(lr_scheduler1)
    # warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=args.warmup_period) # UntunedLinearWarmup for Adam
    
    # create batch iterator
    batch_iterator = iter(train_data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        net.train()
        if iteration != 0 and (iteration % epoch_size == 0):
            loc_loss = 0
            conf_loss = 0
            epoch += 1
            # print(f"epoch={epoch}")
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        # images, targets = next(batch_iterator)
        try:
            images, targets = next(batch_iterator)
        except StopIteration:   
            batch_iterator = iter(train_data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        # with warmup_scheduler.dampening():
        #     for lr_scheduler in sched_list:
        #         lr_scheduler.step()
        #     # lr_scheduler2.step()
        t1 = time.time()
        loc_loss += loss_l.data#[0]
        conf_loss += loss_c.data#[0]

        if iteration % 10 == 0:
            print("iter {:03d} || Loss: {:.4f} || lr: {:.4E} ||".format(int(repr(iteration)),loss.data,get_lr(optimizer)),end=' ')
            print('timer: %.4f sec.' % (t1 - t0))
            # print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data), end=' ')
        # print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')

        # if args.visdom:
        #     update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
        #                     iter_plot, epoch_plot, 'append')

        if iteration != 0 and (((iteration+1) % args.save_iter == 0) or \
                                   ((epoch+1) % args.save_epoch == 0)):
            if (iteration+1) % args.save_iter == 0:
                weight_pth = f'weights/ssd300_{args.dataset}_{iteration+1}.pth'
                save_folder = f'iter_{iteration+1}'
            elif (epoch+1) % args.save_epoch == 0:
                weight_pth = f'weights/ssd300_{args.dataset}_{epoch+1}.pth'
                save_folder = f'epoch_{epoch+1}'

            if os.path.isfile(weight_pth): 
                print("State exists, not saving")
                continue
            else: 
                print('Saving state:',weight_pth)
                torch.save(ssd_net.state_dict(), weight_pth)
                if run_val:
                    val(net,criterion,val_data_loader,val_dataset, args.model_name, save_folder, labelmap)
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# def val(net, criterion, val_data_loader, val_dataset, model_name, save_folder, labelmap):
#     batch_iterator = iter(val_data_loader)
#     net.eval()
#     loc_loss = 0
#     conf_loss = 0
#     average_ap = []
#     # total_objects = 0
#     # total_correct = 0
#     # for i,(images, targets) in enumerate(tqdm(batch_iterator)):
#     with torch.no_grad():
#         for i in tqdm(range(len(val_data_loader))):
#             try:
#                 images, targets = next(batch_iterator)
#             except StopIteration:
#                 batch_iterator = iter(val_data_loader)
#                 images, targets = next(batch_iterator)

#             if args.cuda:
#                 images = Variable(images.cuda())
#                 targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
#             else:
#                 images = Variable(images)
#                 targets = [Variable(ann, volatile=True) for ann in targets]
#             # forward
        
#             t0 = time.time()
#             net.module.set_phase('train')
#             train_output = net(images)
#             # backprop
#             loss_l, loss_c = criterion(train_output, targets)
#             loss = loss_l + loss_c
#             loc_loss += loss_l.data#[0]
#             conf_loss += loss_c.data#[0]

#             # print("out:",train_output[0].shape,train_output[1].shape,train_output[2].shape)
#             # print("targets:{},{}".format(len(targets),targets[0].shape))

#             _, _, w, h = images.shape
#             net.module.set_phase('test')
#             test_output = net(images) # SLOWWWW
#             detections = test_output.data
#             all_boxes = [[[] for _ in range(len(val_dataset))]
#                              for _ in range(len(labelmap)+1)]
#             for j in range(1, detections.size(1)):
#                 dets = detections[0, j, :]
#                 mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
#                 dets = torch.masked_select(dets, mask).view(-1, 5)
#                 if dets.dim() == 0:
#                     continue
#                 boxes = dets[:, 1:]
#                 boxes[:, 0] *= w
#                 boxes[:, 2] *= w
#                 boxes[:, 1] *= h
#                 boxes[:, 3] *= h
#                 scores = dets[:, 0].cpu().numpy()
#                 cls_dets = np.hstack((boxes.cpu().numpy(),
#                                     scores[:, np.newaxis])).astype(np.float32,
#                                                                     copy=False)
#                 all_boxes[j][i] = cls_dets

#             # correct = (out == targets).sum().item()
#             # accuracy = correct / targets.size(0)
#             t1 = time.time()
#             # total_objects += targets.size(0)
#             # total_correct += correct
            
#             output_dir = get_output_dir(model_name, f'{save_folder}/val')
#             class_aps = evaluate_detections(all_boxes, output_dir, dataset_idxs=val_dataset.ids) # NOT that slow!!!
#             average_ap.append(class_aps)

#             # make sure to average across the class, and then together
    
#     print("Val Loss: {:.4f} || Val Loc Loss: {:.4f} || Val Conf Loss: {:.4f} || Val acc: {:.4f} ||".format(loc_loss, 
#                                                                                                            conf_loss,
#                                                                                                            loss.data,
#                                                                                                            0),end=' ') # average_ap/len(batch_iterator)
#     print('timer: %.4f sec.' % (t1 - t0))

#     net.module.set_phase('train')
#     # print(f"Final Accuracy = {total_correct/total_objects}")

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
