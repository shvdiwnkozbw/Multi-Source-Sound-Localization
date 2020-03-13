import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.audioset_data import DataAllocate, AudioVisualData
from model.audioset_detector import Framework, MapLoss, DiscrimLoss, CAMAudio, AlignLoss, CAMVisual, CAMCluster
from utils.logger import AverageMeter, Logger

import json
import cv2
import librosa
import numpy as np
import os
import pickle
from progress.bar import Bar
import argparse
import time
import shutil
import h5py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='AudioVisual Learning')

    parser.add_argument('-d', '--dataset', default='AudioSet', type=str)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')

    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=1, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--val-batch', default=1, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[20, 40],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--warm-up', dest='wp', default=100, type=int,
                        help='warm up learning rate epoch')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--mix', default=4, type=int,
                        help='clips per batch for triplet loss')
    parser.add_argument('--frame', default=4, type=int,
                        help='frames per clip for discrim')
    parser.add_argument('--rois', default=8, type=int,
                        help='rois per frame for visual stream')
    parser.add_argument('--size', default=6, type=int,
                        help='size for roi-align')
    parser.add_argument('--val-per-epoch', '-tp', dest='tp', default=20, type=int,
                        help='number of training epoches between test (default: 30)')
    parser.add_argument('--iter-size', '-is', dest='its', default=1, type=int,
                        help='the forward-backward times within each iteration')
    parser.add_argument('--mode', default='joint', type=str,
                        help='training mode of two-stage, avc for 1st stage, joint for 2nd stage, cls for weak-sup')

    parser.add_argument('--pretrained', '-pre', dest='pre', default=True, type=bool,
                        help='whether to use pretrained model')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    return args

args = parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_gpu = torch.cuda.is_available() and int(args.gpu_id) >= 0

def main():
    start_epoch = args.start_epoch
    max_acc = 0.0

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    print('Preparing Dataset %s' % args.dataset)

    trainset = AudioVisualData(mix=args.mix, frame=args.frame, training=True)
    valset = AudioVisualData(mix=args.mix, frame=args.frame, training=False)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=False,
                                  num_workers=args.workers, collate_fn=DataAllocate)
    valloader = data.DataLoader(valset, batch_size=args.val_batch, shuffle=False,
                                num_workers=args.workers, collate_fn=DataAllocate)

    framework = Framework(args.pre, args.mix, args.frame)
    discrmloss = DiscrimLoss()
    maploss = MapLoss()
    camaudio = CAMAudio(framework)
    camvisual = CAMVisual(framework)
    camcluster = CAMCluster(framework)
    alignloss = AlignLoss()

    for p in framework.visual.conv1.parameters():
        p.requires_grad = False
    for p in framework.visual.bn1.parameters():
        p.requires_grad = False
    for p in framework.visual.layer1.parameters():
        p.requires_grad = False
    for p in framework.audio.features[0].parameters():
        p.requires_grad = False

    if use_gpu:
        framework = framework.cuda()
        discrmloss = discrmloss.cuda()
        maploss = maploss.cuda()
        alignloss = alignloss.cuda()
        camaudio = camaudio.cuda()
        camvisual = camvisual.cuda()
        camcluster = camcluster.cuda()

    head_params = list(map(id, framework.discrim.parameters()))
    head_params += list(map(id, framework.audio.outputlayer.parameters()))
    head_params += list(map(id, framework.visual.fc.parameters()))
    head_params += list(map(id, framework.avalign.parameters()))

    backbone_params = filter(lambda x: id(x) not in head_params, framework.parameters())
    head_params = filter(lambda x: id(x) in head_params, framework.parameters())

    optimizer = optim.SGD([{'params': head_params, 'lr': args.lr},
                           {'params': backbone_params, 'lr': 0.1*args.lr}],
                          lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # optimizer = optim.Adam([{'params': head_params, 'lr': args.lr},
    #                        {'params': backbone_params, 'lr': 0.0001}],
    #                        lr=args.lr, weight_decay=args.weight_decay)

    args.checkpoint = os.path.join(args.checkpoint, args.dataset)
    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoint)

    if args.resume:
        logger = Logger(os.path.join(args.checkpoint, 'audioset.txt'), title='AudioVisual Learning', resume=True)
        checkpoint = torch.load(args.resume)
        framework.load_state_dict(checkpoint['state_dict'], strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title='AudioVisual Learning', resume=False)
        logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'A2V', 'V2A'])

    if args.evaluate:
#        extract(valloader, camcluster, discrmloss, start_epoch, use_gpu)
        test_cam(valloader, framework, [camaudio, camvisual], start_epoch, use_gpu)
#        a2v, v2a = test_avs(valloader, framework, start_epoch, use_gpu)
#        print('Accuracy: ', a2v, v2a)
        logger.close()
        return

    for epoch in range(start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)
        if epoch == 0:
            warm_up_lr(optimizer, True)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, args.lr))

        if args.mode == 'avc':
            loss = train(trainloader, framework, [camaudio, camvisual], discrmloss, maploss,
                         alignloss, optimizer, epoch, use_gpu)
        if args.mode == 'joint':
            loss = train(trainloader, framework, [camaudio, camvisual], discrmloss, maploss,
                         alignloss, optimizer, epoch, use_gpu)

        if (epoch + 1) % args.tp == 0:
            a2v, v2a = test_avs(valloader, framework, start_epoch, use_gpu)
            logger.append([epoch, args.lr, loss, a2v, v2a])
        else:
            a2v = 0.0
            v2a = 0.0
            logger.append([epoch, args.lr, loss, a2v, v2a])

        is_best = ((a2v + v2a) > max_acc)
        max_acc = max(max_acc, (a2v + v2a))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': framework.state_dict(),
            'loss': loss,
            'optimizer': optimizer.state_dict(),
            'accuracy': (a2v + v2a)
        }, epoch + 1, is_best, checkpoint=args.checkpoint, filename='model_'+args.mode)

    logger.close()

def train_avs(trainloader, model, cam, discrimloss, maploss, alignloss, optimizer, epoch, use_gpu):
    model.eval()
    model.audio.gru.train()
    model.discrim.train()
    discrimloss.train()
    maploss.train()
    alignloss.train()
    camaudio = cam[0]
    camvisual = cam[1]

    data_time = AverageMeter()
    a_loss = AverageMeter()
    v_loss = AverageMeter()
    dis_loss = AverageMeter()
    total_loss = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    optimizer.zero_grad()

    weight = 1.0 if epoch > 0 else 0.0

    for batch_idx, (audio, visual, label) in enumerate(trainloader):

        audio = audio.view(audio.shape[0] * args.mix, *audio.shape[-2:])
        visual = visual.view(visual.shape[0] * args.mix * args.frame, *visual.shape[-3:])
        label = label.view(label.shape[0] * args.mix, label.shape[-1])

        if batch_idx == args.wp:
            warm_up_lr(optimizer, False)

        data_time.update(time.time() - end)
        if use_gpu:
            audio = audio.cuda()
            visual = visual.cuda()
            label = label.cuda()

        data_time.update(time.time() - end)
        discrim, pred_a, pred_v, feat_a, feat_v, cam_v = model(audio, visual)
        dloss = discrimloss(discrim[0], discrim[1])
        aloss, vloss = maploss(label, pred_a, pred_v)
        loss = aloss / float(15) + vloss / float(15) + dloss

        if loss.item() > 0:
            total_loss.update(loss.item(), 1)
            a_loss.update(aloss.item() / 15, 1)
            v_loss.update(vloss.item() / 15, 1)
            dis_loss.update(dloss.item(), 1)

        loss /= args.its
        loss.backward()

        if batch_idx % args.its == 0:
            optimizer.step()
            optimizer.zero_grad()

        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s |Loss: {loss:.3f} |ALoss: {aloss:.3f} ' \
                     '|VLoss: {vloss:.3f} |DLoss: {dloss:.3f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.val,
            loss=total_loss.val,
            aloss=a_loss.val,
            vloss=v_loss.val,
            dloss=dis_loss.val
        )
        bar.next()

    bar.finish()

    return total_loss.avg

def train_cls(trainloader, model, cam, discrimloss, maploss, alignloss, optimizer, epoch, use_gpu):
    model.eval()
    model.audio.gru.train()
    maploss.train()
    alignloss.train()
    camaudio = cam[0]
    camvisual = cam[1]

    data_time = AverageMeter()
    a_loss = AverageMeter()
    v_loss = AverageMeter()
    total_loss = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    optimizer.zero_grad()

    for batch_idx, (audio, visual, label) in enumerate(trainloader):

        audio = audio.view(audio.shape[0] * args.mix, *audio.shape[-2:])
        visual = visual.view(visual.shape[0] * args.mix * args.frame, *visual.shape[-3:])
        label = label.view(label.shape[0] * args.mix, label.shape[-1])

        if batch_idx == args.wp:
            warm_up_lr(optimizer, False)

        data_time.update(time.time() - end)
        if use_gpu:
            audio = audio.cuda()
            visual = visual.cuda()
            label = label.cuda()

        data_time.update(time.time() - end)
        pred_a, pred_v = model(audio, visual)
        aloss, vloss = maploss(label, pred_a, pred_v)
        loss = aloss / float(15) + vloss / float(15)

        if loss.item() > 0:
            total_loss.update(loss.item(), 1)
            a_loss.update(aloss.item() / 15, 1)
            v_loss.update(vloss.item() / 15, 1)

        loss /= args.its
        loss.backward()

        if batch_idx % args.its == 0:
            optimizer.step()
            optimizer.zero_grad()

        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s |Loss: {loss:.3f} |ALoss: {aloss:.3f} ' \
                     '|VLoss: {vloss:.3f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.val,
            loss=total_loss.val,
            aloss=a_loss.val,
            vloss=v_loss.val
        )
        bar.next()

    bar.finish()

    return total_loss.avg

def train(trainloader, model, cam, discrimloss, maploss, alignloss, optimizer, epoch, use_gpu):
    model.eval()
    model.audio.gru.train()
    model.discrim.train()
    discrimloss.train()
    maploss.train()
    alignloss.train()
    camaudio = cam[0]
    camvisual = cam[1]

    data_time = AverageMeter()
    a_loss = AverageMeter()
    v_loss = AverageMeter()
    e_loss = AverageMeter()
    dis_loss = AverageMeter()
    total_loss = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    optimizer.zero_grad()

    for batch_idx, (audio, visual, label) in enumerate(trainloader):

        audio = audio.view(audio.shape[0] * args.mix, *audio.shape[-2:])
        visual = visual.view(visual.shape[0] * args.mix * args.frame, *visual.shape[-3:])
        label = label.view(label.shape[0] * args.mix, label.shape[-1])

        if batch_idx == args.wp:
            warm_up_lr(optimizer, False)

        data_time.update(time.time() - end)
        if use_gpu:
            audio = audio.cuda()
            visual = visual.cuda()
            label = label.cuda()

        data_time.update(time.time() - end)
        discrim, pred_a, pred_v, feat_a, feat_v, cam_v = model(audio, visual)
        feat_a, _, _ = camaudio(pred_a, feat_a)
        # feat_v = camvisual(cam_v, feat_v)
        cam_v = camvisual(cam_v, feat_v)
        common, differ = model.avalign(feat_a, feat_v, label, cam_v)
        eloss = alignloss(common, differ)
        dloss = discrimloss(discrim[0], discrim[1])
        aloss, vloss = maploss(label, pred_a, pred_v)
        loss = aloss / float(15) + vloss / float(15) + eloss + dloss

        if loss.item() > 0:
            total_loss.update(loss.item(), 1)
            a_loss.update(aloss.item() / 15, 1)
            v_loss.update(vloss.item() / 15, 1)
            e_loss.update(eloss.item(), 1)
            dis_loss.update(dloss.item(), 1)

        loss /= args.its
        loss.backward()

        if batch_idx % args.its == 0:
            optimizer.step()
            optimizer.zero_grad()

        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s |Loss: {loss:.3f} |ALoss: {aloss:.3f} ' \
                     '|VLoss: {vloss:.3f} |DLoss: {dloss:.3f} |ELoss: {eloss:.3f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.val,
            loss=total_loss.val,
            aloss=a_loss.val,
            vloss=v_loss.val,
            dloss=dis_loss.val,
            eloss=e_loss.val
        )
        bar.next()

    bar.finish()

    return total_loss.avg

def train_cam(trainloader, model, cam, discrimloss, maploss, alignloss, optimizer, epoch, use_gpu):
    model.eval()
    model.audio.gru.train()
    model.discrim.train()
    discrimloss.train()
    maploss.train()
    alignloss.train()
    camaudio = cam[0]
    camvisual = cam[1]

    data_time = AverageMeter()
    a_loss = AverageMeter()
    v_loss = AverageMeter()
    e_loss = AverageMeter()
    dis_loss = AverageMeter()
    total_loss = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    optimizer.zero_grad()

    for batch_idx, (audio, visual, label) in enumerate(trainloader):

        audio = audio.view(audio.shape[0] * args.mix, *audio.shape[-2:])
        visual = visual.view(visual.shape[0] * args.mix * args.frame, *visual.shape[-3:])
        label = label.view(label.shape[0] * args.mix, label.shape[-1])

        if batch_idx == args.wp:
            warm_up_lr(optimizer, False)

        data_time.update(time.time() - end)
        if use_gpu:
            audio = audio.cuda()
            visual = visual.cuda()
            label = label.cuda()

        data_time.update(time.time() - end)
        pred_a, pred_v, feat_a, feat_v, cam_v = model(audio, visual)
        feat_a, _, _ = camaudio(pred_a, feat_a)
        cam_v = camvisual(cam_v, feat_v)
        common, differ = model.avalign(feat_a, feat_v, label, cam_v)
        eloss = alignloss(common, differ)
        aloss, vloss = maploss(label, pred_a, pred_v)
        loss = aloss / float(15) + vloss / float(15) + eloss

        if loss.item() > 0:
            total_loss.update(loss.item(), 1)
            a_loss.update(aloss.item() / 15, 1)
            v_loss.update(vloss.item() / 15, 1)
            e_loss.update(eloss.item(), 1)

        loss /= args.its
        loss.backward()

        if batch_idx % args.its == 0:
            optimizer.step()
            optimizer.zero_grad()

        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s |Loss: {loss:.3f} |ALoss: {aloss:.3f} ' \
                     '|VLoss: {vloss:.3f} |ELoss: {eloss:.3f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.val,
            loss=total_loss.val,
            aloss=a_loss.val,
            vloss=v_loss.val,
            eloss=e_loss.val
        )
        bar.next()

    bar.finish()

    return total_loss.avg

def test_avs(valloader, model, epoch, use_gpu):
    model.eval()
    # model.discrim.train()
    data_time = AverageMeter()
    match_rate = AverageMeter()
    differ_rate = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=len(valloader))

    for batch_idx, (audio, visual, label) in enumerate(valloader):

        audio = audio.view(audio.shape[0] * args.mix, *audio.shape[-2:])
        visual = visual.view(visual.shape[0] * args.mix * args.frame, *visual.shape[-3:])
        label = label.view(label.shape[0] * args.mix, label.shape[-1])

        data_time.update(time.time() - end)
        if use_gpu:
            audio = audio.cuda()
            visual = visual.cuda()
            label = label.cuda()

        data_time.update(time.time() - end)
        discrim, pred_a, pred_v, feat_a, feat_v, cam_v = model(audio, visual)
        common = discrim[0].view(-1, 2)
        differ = discrim[1].view(-1, 2)
        true_match = torch.sum(common[:, 1] > common[:, 0])
        true_match = true_match.item() / float(common.shape[0])
        true_differ = torch.sum(differ[:, 0] > differ[:, 1])
        true_differ = true_differ.item() / float(differ.shape[0])
        match_rate.update(true_match, 1)
        differ_rate.update(true_differ, 1)

        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s |Match: {match:.3f} |Differ: {differ: .3f}'.format(
            batch=batch_idx + 1,
            size=len(valloader),
            data=data_time.val,
            match=match_rate.val,
            differ=differ_rate.val
        )
        bar.next()

    bar.finish()
    return match_rate.avg, differ_rate.avg

def test_cls(valloader, model, epoch, use_gpu):
    model.eval()
    # model.discrim.train()
    data_time = AverageMeter()
    match_rate = AverageMeter()
    differ_rate = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=len(valloader))
    infermap = np.zeros((1098, 10, 15, 16, 16))
    effidx = np.load('audioset_val.npy')
    result = []

    for batch_idx, (audio, visual, label) in enumerate(valloader):

        audio = audio.view(audio.shape[0] * args.mix, *audio.shape[-2:])
        visual = visual.view(visual.shape[0] * args.mix * args.frame, *visual.shape[-3:])
        label = label.view(label.shape[0] * args.mix, label.shape[-1])

        data_time.update(time.time() - end)
        if use_gpu:
            audio = audio.cuda()
            visual = visual.cuda()
            label = label.cuda()

        data_time.update(time.time() - end)
        pred_a, pred_v = model(audio, visual)
        visual_ = model.visual(visual, 'bottom')
        visual_ = model.visual.layer4(visual_)
        visual_ = visual_.view(10, 512, 256)
        pred = torch.matmul(model.visual.fc.weight.unsqueeze(0), visual_)
        pred = pred.view(10, 15, 16, 16)
        # pred = torch.nn.functional.interpolate(pred, (256, 256), mode='bilinear')
        # pred = torch.relu(pred)
        pred = pred.detach().cpu().numpy()
        infermap[batch_idx] = pred
        # generate_bbox(batch_idx, pred_a, pred_v, pred, effidx, result)
        # cam_visualize(batch_idx, visual, pred)
        match = (pred_a>0.5) * (label>0)
        differ = (pred_a<0.5) * (label==0)
        if torch.sum(label) > 0:
            match_rate.update(torch.sum(match).item() / float(torch.sum(label)))
        differ_rate.update(torch.sum(differ).item() / float(torch.sum(label==0)))

        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s |Match: {match:.3f} |Differ: {differ: .3f}'.format(
            batch=batch_idx + 1,
            size=len(valloader),
            data=data_time.val,
            match=match_rate.val,
            differ=differ_rate.val
        )
        bar.next()

    bar.finish()
    np.save('infer', infermap)
    # with open('bbox.json', 'w') as f:
    #     json.dump(result, f)

    return match_rate.avg, differ_rate.avg

def test_cam(valloader, model, cam, epoch, use_gpu):
    model.eval()
    model.audio.gru.train()
    model.discrim.train()
    data_time = AverageMeter()
    end = time.time()
    camaudio = cam[0]
    camvisual = cam[1]
    infermap = np.zeros((1098, 10, 15, 16, 16))
    effidx = np.load('audioset_val.npy')
    result = []
    bar = Bar('Processing', max=len(valloader))

    for batch_idx, (audio, visual, label) in enumerate(valloader):

        audio = audio.view(audio.shape[0] * args.mix, *audio.shape[-2:])
        visual = visual.view(visual.shape[0] * args.mix * args.frame, *visual.shape[-3:])
        label = label.view(label.shape[0] * args.mix, label.shape[-1])

        data_time.update(time.time() - end)
        if use_gpu:
            audio = audio.cuda()
            visual = visual.cuda()
            label = label.cuda()

        data_time.update(time.time() - end)
        # discrim, pred_a, pred_v, feat_a, feat_v, cam_v = model(audio, visual)
        pred_a, pred_v, feat_a, feat_v, cam_v = model(audio, visual, False)
        feat_a, _, _ = camaudio(pred_a, feat_a)
        feat_a = model.discrim.temp_conv(feat_a)
        feat_a = model.avalign.temp_pool(feat_a)
        cam_v = camvisual(cam_v, feat_v)
        cam_v = torch.max(cam_v, 1)[0]
        feat_v = model.discrim.spa_conv(feat_v)
        feat_a = feat_a.view(-1, 1, 15, feat_a.shape[1], 1)
        feat_v = feat_v.view(-1, args.frame, 1, feat_v.shape[1],
                             feat_v.shape[-2]*feat_v.shape[-1])
        feat_a = feat_a.repeat([1, args.frame, 1, 1, feat_v.shape[-1]])
        feat_v = feat_v.repeat([1, 1, 15, 1, 1])
        feat = torch.cat([feat_a.permute(0, 1, 2, 4, 3).contiguous(),
                          feat_v.permute(0, 1, 2, 4, 3).contiguous()], -1)
#        score = model.discrim.discrim(feat)
#        score = torch.softmax(score, -1)[:, :, :, :, 1]
#        score = score.view(-1, 15, 16, 16)
        embed = feat.view(-1, 1024)
        dist = F.mse_loss(model.discrim.transform_a(embed[:, :512]),
                          model.discrim.transform_a(embed[:, 512:]), reduce=False)
        dist = dist.mean(-1)
        dist = dist.view(*feat.shape[:4])
        score = torch.softmax(-dist, -1) * cam_v.view(1, 10, 1, 256)
        score = score / torch.max(score+1e-10, -1)[0].unsqueeze(-1)
        score = score.view(-1, 15, 16, 16)
        score = torch.nn.functional.interpolate(score, (256, 256), mode='bilinear')
        score = score.detach().cpu().numpy()
        infermap[batch_idx] = score
        # generate_bbox(batch_idx, pred_a, pred_v, score, effidx, result)
        # cam_visualize(batch_idx, visual, score)

        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s'.format(
            batch=batch_idx + 1,
            size=len(valloader),
            data=data_time.val
        )
        bar.next()

    bar.finish()
    np.save('infer', infermap)
    # with open('bbox.json', 'w') as f:
    #     json.dump(result, f)

def extract(trainloader, model, discrimloss, epoch, use_gpu):
    model.model.eval()
    model.model.discrim.train()
    model.model.audio.gru.train()

    data_time = AverageMeter()
    dis_loss = AverageMeter()
    end = time.time()
    infer = np.zeros((1098, 10, 16, 16))

    bar = Bar('Processing', max=len(trainloader))

    for batch_idx, (audio, visual, _) in enumerate(trainloader):

        audio = audio.view(audio.shape[0] * args.mix, *audio.shape[-2:])
        visual = visual.view(visual.shape[0] * args.mix * args.frame, *visual.shape[-3:])

        data_time.update(time.time() - end)
        if use_gpu:
            audio = audio.cuda()
            visual = visual.cuda()

        data_time.update(time.time() - end)
        discrim, cam = model(audio, visual)
        # cam_visualize(batch_idx, visual, cam)
        cam = cam.detach().cpu().numpy()
        infer[batch_idx] = cam

        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.val
        )
        bar.next()

    bar.finish()
    np.save('infer', infer)

def l2norm(x):
    norm = np.sum(x * x, -1)
    norm = np.sqrt(norm)
    norm = np.max(norm)
    return norm


def cam_visualize(batch_idx, visual, cam):
    assert visual.shape[0] == cam.shape[0]
    segments = visual.shape[0]
    visual = visual.cpu().numpy()
    cam = cam.detach().cpu().numpy()
    visual = visual.transpose([0, 2, 3, 1])
    visual = visual * np.array([0.229, 0.224, 0.225])
    visual = visual + np.array([0.485, 0.456, 0.406])
    visual = np.clip(visual, 0.0, 1.0)
    for seg in range(segments):
        if len(cam.shape) == 4:
            for i in range(15):
                actmap = cam[seg][i]
                actmap = actmap / np.max(actmap+1e-10)
                actmap = actmap * 255
                actmap = actmap.astype(np.uint8)
                actmap = cv2.applyColorMap(actmap, cv2.COLORMAP_JET)[:, :, ::-1]
                actmap = actmap / 255.0
                img = visual[seg]
                plt.imsave('./vis/' + str(batch_idx) + '_' + str(15*seg+i) + '.jpg',
                           0.5 * actmap + 0.5 * img)
        else:
            actmap = cv2.resize(cam[seg], (256, 256))
            actmap = actmap * 255
            actmap = cv2.applyColorMap(actmap.astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
            actmap = actmap / 255.0
            img = visual[seg]
            plt.imsave('./vis/' + str(batch_idx) + '_' + str(seg) + '.jpg', 0.5 * actmap + 0.5 * img)


def save_checkpoint(state, epoch, is_best, checkpoint='checkpoint', filename='checkpoint'):
    filepath = os.path.join(checkpoint, filename + '.pth.tar')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, filename + '_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        args.lr *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr


def warm_up_lr(optimizer, warm_up):
    if warm_up:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1 * args.lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

if __name__ == '__main__':
    main()
