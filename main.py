import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.dataset import DataAllocate, AudioVisualData
from model.detector import Framework, MapLoss, DiscrimLoss, StatFreq, CAMAudio, AlignLoss, CAMCluster, CAMVisual
from utils.logger import AverageMeter, Logger

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

    parser.add_argument('-d', '--dataset', default='Flickr', type=str)
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')

    parser.add_argument('--epochs', default=60, type=int, metavar='N',
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
    parser.add_argument('--val-per-epoch','-tp', dest='tp', default=2, type=int,
                        help='number of training epoches between test (default: 30)')
    parser.add_argument('--iter-size', '-is', dest='its', default=4, type=int,
                        help='the forward-backward times within each iteration')
    parser.add_argument('--mode', default='joint', type=str,
                        help='training mode of two-stage, avc for 1st stage, joint for 2nd stage')

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

def open_file(datadir):
    audio = os.path.join(datadir, 'Spec.h5')
    video = os.path.join(datadir, 'Video.h5')
    rois = os.path.join(datadir, 'Roi.h5')
    return audio, video, rois

def main():

    start_epoch = args.start_epoch
    max_acc = 0.0
    
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    
    print('Preparing Dataset %s' % args.dataset)
    
    datadir = '/media/yuxi/Data/AVEh5/' if args.dataset == 'AVE_C' else '/media/yuxi/Data/AudioVisual/data/Flickr/'
    audio, video, rois = open_file(datadir)

    trainset = AudioVisualData(audio, video, rois, mix=args.mix, frame=args.frame, dataset=args.dataset, training=True)
    valset = AudioVisualData(audio, video, rois, mix=args.mix, frame=args.frame, dataset=args.dataset, training=False)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers,
                                  collate_fn=DataAllocate)
    valloader = data.DataLoader(valset, batch_size=args.val_batch, shuffle=False, num_workers=args.workers, 
                                collate_fn=DataAllocate)
    
    framework = Framework(args.pre, args.mix, args.frame, args.rois, args.size)
    discrmloss = DiscrimLoss(2)
    maploss = MapLoss()
    cluster = CAMCluster(framework)
    camaudio = CAMAudio(framework)
    camvisual = CAMVisual(framework)
    alignloss = AlignLoss(2)
    stat = StatFreq(527, 1000)

    for p in framework.visual.conv1.parameters():
        p.requires_grad = False
    for p in framework.visual.bn1.parameters():
        p.requires_grad = False
    for p in framework.visual.layer1.parameters():
        p.requires_grad = False
    for p in framework.audio.features[0].parameters():
        p.requires_grad = False
    for p in framework.prior_a.parameters():
        p.requires_grad = False
    for p in framework.prior_v.parameters():
        p.requires_grad = False

    cluster_a = np.load('utils/cluster_a.npy')
    cluster_v = np.load('utils/cluster_v3.npy')
    cluster_a = torch.FloatTensor(cluster_a.T)
    cluster_v = torch.FloatTensor(cluster_v)

    if use_gpu:
        framework = framework.cuda()
        discrmloss = discrmloss.cuda()
        maploss = maploss.cuda()
        alignloss = alignloss.cuda()
        camaudio = camaudio.cuda()
        camvisual = camvisual.cuda()
        cluster = cluster.cuda()
        stat = stat.cuda()
        cluster_a = cluster_a.cuda()
        cluster_v = cluster_v.cuda()

    # for p in framework.parameters():
    #     p.requires_grad = False
    #
    # statist(trainloader, framework, cluster_a, cluster_v, stat, use_gpu)
    # return

    head_params = list(map(id, framework.discrim.parameters()))
    head_params += list(map(id, framework.audio.outputlayer.parameters()))
    head_params += list(map(id, framework.visual.fc.parameters()))
    head_params += list(map(id, framework.avalign.parameters()))

    backbone_params = filter(lambda x: id(x) not in head_params, framework.parameters())  
    head_params = filter(lambda x: id(x) in head_params, framework.parameters())
    
    # optimizer = optim.SGD([{'params': head_params, 'lr': args.lr},
    #                        {'params': backbone_params, 'lr': 0.0001}],
    #                       lr=args.lr, momentum=args.momentum,
    #                       weight_decay=args.weight_decay)
    
    optimizer = optim.Adam([{'params': head_params, 'lr': args.lr},
                           {'params': backbone_params, 'lr': 0.0001}],
                           lr=args.lr, weight_decay=args.weight_decay)
    
    args.checkpoint = os.path.join(args.checkpoint, args.dataset)
    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoint)
    
    if args.resume:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title='AudioVisual Learning', resume=True)
        checkpoint = torch.load(args.resume)
        framework.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title='AudioVisual Learning', resume=False)
        logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'A2V', 'V2A'])
    
    if args.evaluate:
        # extract(valloader, cluster, discrmloss, start_epoch, use_gpu)
        a2v, v2a = test_avs(valloader, framework, cluster_a, cluster_v, start_epoch, use_gpu)
        print('Accuracy: ', a2v, v2a)
        logger.close()
        return
    
    for epoch in range(start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch)        
        if epoch == 0:
            warm_up_lr(optimizer, True)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, args.lr))
        
        if args.mode == 'joint':
            loss = train(trainloader, framework, [camaudio, camvisual], cluster_a, cluster_v, discrmloss, maploss, alignloss, optimizer, epoch, use_gpu)
        if args.mode == 'avc':
            loss = train_avs(trainloader, framework, cluster_a, cluster_v, discrmloss, maploss, optimizer, epoch, use_gpu)

        if (epoch+1) % args.tp == 0:
            # a2v = test_a2v(valloader, framework, cluster_a, cluster_v, start_epoch, use_gpu)
            # v2a = test_v2a(valloader, framework, cluster_a, cluster_v, start_epoch, use_gpu)
            a2v, v2a = test_avs(valloader, framework, cluster_a, cluster_v, epoch, use_gpu)
            logger.append([epoch, args.lr, loss, a2v, v2a])
        else:
            a2v = 0.0
            v2a = 0.0
            logger.append([epoch, args.lr, loss, a2v, v2a])
        
        is_best = ((a2v+v2a) > max_acc)
        max_acc = max(max_acc, (a2v+v2a))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': framework.state_dict(),
            'loss': loss,
            'optimizer': optimizer.state_dict(),
            'accuracy': (a2v+v2a)
        }, epoch + 1, is_best, checkpoint=args.checkpoint, filename='model_'+args.mode)

    logger.close()
    

def train(trainloader, model, cam, cluster_a, cluster_v, discrimloss, maploss, alignloss, optimizer, epoch, use_gpu):
    
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
    r_loss = AverageMeter()
    e_loss = AverageMeter()
    dis_loss = AverageMeter()
    total_loss = AverageMeter()
    end = time.time()
    
    bar = Bar('Processing', max=len(trainloader))
    optimizer.zero_grad()

    for batch_idx, (audio, visual, roi) in enumerate(trainloader):
        
        audio = audio.view(audio.shape[0]*args.mix, *audio.shape[-2:])
        visual = visual.view(visual.shape[0]*args.mix*args.frame, *visual.shape[-3:])
        roi = roi.view(roi.shape[0], args.mix*args.frame, args.rois, 4)
        
        if  batch_idx == args.wp:
            warm_up_lr(optimizer, False)

        data_time.update(time.time() - end)
        if use_gpu:
            audio = audio.cuda()
            visual = visual.cuda()
            roi = roi.cuda()

        data_time.update(time.time() - end)
        discrim, mask, pred_a, pred_v, label_a, label_v, feat, cam_v = model(audio, visual, roi, cluster_a, cluster_v)
        feat_a, _, _ = camaudio(pred_a, feat[0])
        cam_v = camvisual(cam_v, feat[1])
        common, differ = model.avalign(feat_a, pred_a, feat[1], pred_v[0], cam_v)
        eloss = alignloss(common, differ)
        dloss = discrimloss(discrim[0], discrim[1], mask)
        aloss, vloss, rloss = maploss(label_a, *label_v, pred_a, *pred_v)
        loss = aloss/float(7) + vloss/float(7) + eloss + rloss + dloss

        if loss.item() > 0:
            total_loss.update(loss.item(), 1)
            a_loss.update(aloss.item()/7, 1)
            v_loss.update(vloss.item()/7, 1)
            r_loss.update(rloss.item()/7, 1)
            e_loss.update(eloss.item(), 1)
            dis_loss.update(dloss.item(), 1)
            
        loss /= args.its
        loss.backward()

        if batch_idx % args.its == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        end = time.time()
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s |Loss: {loss:.3f} |ALoss: {aloss:.3f} |VLoss: {vloss:.3f} |RLoss: {rloss: .3f} |DLoss: {dloss:.3f} |ELoss: {eloss:.3f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.val,
            loss=total_loss.val,
            aloss=a_loss.val,
            vloss=v_loss.val,
            rloss=r_loss.val,
            dloss=dis_loss.val,
            eloss=e_loss.val
        )
        bar.next()
        
    bar.finish()
    
    return total_loss.avg


def train_avs(trainloader, model, cluster_a, cluster_v, discrimloss, maploss, optimizer, epoch, use_gpu):
    model.eval()
    model.audio.gru.train()
    model.visual.layer5.train()
    model.discrim.train()
    discrimloss.train()
    maploss.train()

    data_time = AverageMeter()
    a_loss = AverageMeter()
    v_loss = AverageMeter()
    r_loss = AverageMeter()
    dis_loss = AverageMeter()
    total_loss = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    optimizer.zero_grad()

    for batch_idx, (audio, visual, roi) in enumerate(trainloader):

        audio = audio.view(args.train_batch * args.mix, *audio.shape[-2:])
        visual = visual.view(args.train_batch * args.mix * args.frame, *visual.shape[-3:])
        roi = roi.view(args.train_batch, args.mix * args.frame, args.rois, 4)

        if batch_idx == args.wp:
            warm_up_lr(optimizer, False)

        data_time.update(time.time() - end)
        if use_gpu:
            audio = audio.cuda()
            visual = visual.cuda()
            roi = roi.cuda()

        data_time.update(time.time() - end)
        discrim, mask, pred_a, pred_v, label_a, label_v, feat, _ = model(audio, visual, roi, cluster_a, cluster_v)
        dloss = discrimloss(discrim[0], discrim[1], mask)
        aloss, vloss, rloss = maploss(label_a, *label_v, pred_a, *pred_v)
        loss = dloss + vloss / float(7) + rloss + aloss

        if loss.item() > 0:
            total_loss.update(loss.item(), 1)
            a_loss.update(aloss.item() / 7, 1)
            v_loss.update(vloss.item() / 7, 1)
            r_loss.update(rloss.item(), 1)
            dis_loss.update(dloss.item(), 1)

        loss /= args.its
        loss.backward()

        if batch_idx % args.its == 0:
            optimizer.step()
            optimizer.zero_grad()

        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s |Loss: {loss:.3f} |ALoss: {aloss:.3f} |VLoss: {vloss:.3f} |RLoss: {rloss: .3f} |DLoss: {dloss:.3f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.val,
            loss=total_loss.val,
            aloss=a_loss.val,
            vloss=v_loss.val,
            rloss=r_loss.val,
            dloss=dis_loss.val
        )
        bar.next()

    bar.finish()

    return total_loss.avg

def extract(trainloader, model, discrimloss, epoch, use_gpu):
    model.model.eval()
    model.model.audio.gru.train()

    data_time = AverageMeter()
    dis_loss = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))

    for batch_idx, (audio, visual, _) in enumerate(trainloader):

        audio = audio.view(args.val_batch * args.mix, *audio.shape[-2:])
        visual = visual.view(args.val_batch * args.mix * args.frame, *visual.shape[-3:])

        data_time.update(time.time() - end)
        if use_gpu:
            audio = audio.cuda()
            visual = visual.cuda()

        data_time.update(time.time() - end)
        discrim, cam = model(audio, visual)
        dloss = discrimloss(discrim[0], discrim[1], None)
        cam_visualize(batch_idx, visual, cam)

        if dloss.item() > 0:
            dis_loss.update(dloss.item(), 1)

        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s |DLoss: {dloss:.3f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.val,
            dloss=dis_loss.val
        )
        bar.next()

    bar.finish()

def test_avs(valloader, model, cluster_a, cluster_v, epoch, use_gpu):
    
    model.eval()
    data_time = AverageMeter()
    match_rate = AverageMeter()
    differ_rate = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=len(valloader))

    for batch_idx, (audio, visual, roi) in enumerate(valloader):

        audio = audio.view(args.val_batch * args.mix, *audio.shape[-2:])
        visual = visual.view(args.val_batch * args.mix * args.frame, *visual.shape[-3:])
        roi = roi.view(args.val_batch, args.mix * args.frame, args.rois, 4)

        data_time.update(time.time() - end)
        if use_gpu:
            audio = audio.cuda()
            visual = visual.cuda()
            roi = roi.cuda()

        data_time.update(time.time() - end)
        discrim, mask, pred_a, pred_v, label_a, label_v, _, _ = model(audio, visual, roi, cluster_a, cluster_v)
        discrim = [discrim[0].view(-1, 1), discrim[1].view(-1, 1)]
        discrim = torch.cat(discrim, 0)
        segments = discrim.shape[0]
        true_match = torch.sum(discrim[: segments//2, 0]<1.0)
        true_match = true_match.item() / float(segments/2)
        true_differ = torch.sum(discrim[segments//2:, 0]>1.0)
        true_differ = true_differ.item() / float(segments/2)
        match_rate.update(true_match, 1)
        differ_rate.update(true_differ, 1)

        end = time.time()
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s |Match: {match:.3f} |Differ: {differ: .3f}'.format(
            batch=batch_idx + 1,
            size=len(valloader),
            data=data_time.val,
            match=match_rate.val,
            differ=differ_rate.val
        )
        bar.next()
        
    bar.finish()
    return match_rate.avg, differ_rate.avg

def l2norm(x):
    norm = np.sum(x*x, -1)
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
        actmap = cv2.resize(cam[seg], (256, 256))
        actmap = np.expand_dims(actmap, -1)
        img = visual[seg]
        plt.imsave('./vis/'+str(batch_idx)+'_'+str(seg)+'.jpg', 0.6*actmap+0.4*img)

def visualize(batch_idx, prefix, visual, rois, spa_att, roi_att):
#    visual (batchsize*mix, 3, w, h)
#    rois (batchsize, mix, rois, 4)
#    spa_att (batchsize, mix, rois, 1, 9)
#    roi_att (batchsize, mix, rois)
    batchsize = rois.shape[0]
    mix = rois.shape[1]
    num_rois = rois.shape[2]
    spa_att = spa_att.reshape([batchsize*mix, num_rois, args.size*args.size])
    roi_att = roi_att.reshape([batchsize*mix, num_rois, 1])
    attention = spa_att * roi_att
    attention = attention.reshape([batchsize*mix, num_rois, args.size, args.size])
    attention = attention.cpu()
    rois = rois.reshape([batchsize*mix, num_rois, 4])
    rois = rois.cpu().numpy()
    img = visual.cpu().numpy()
    img = img.transpose([0, 2, 3, 1])
    img = img * np.array([0.229, 0.224, 0.225])
    img = img + np.array([0.485, 0.456, 0.406])
    img = img * 255.0
    for i in range(len(img)):
        att = cal_att(img, rois, attention, i)
        cv2.imwrite(str(batch_idx)+str(i)+'.jpg', att)

def cal_att(imgs, rois, attention, idx):
    attmap = np.zeros((256, 256))
    att = attention[idx]
    img = imgs[idx]
    roi = rois[idx]
    for i in range(roi.shape[0]):
        xmin, ymin, xmax, ymax = np.int16(roi[i])
        w = xmax - xmin
        h = ymax - ymin
        weight = torch.nn.functional.interpolate(att[i].reshape([1, 1, args.size, args.size]), size=(h, w), mode='bilinear')
        weight = weight[0, 0]
        attmap[ymin: ymax, xmin: xmax] = weight.detach().numpy() + attmap[ymin: ymax, xmin: xmax]
    attmap = (attmap-np.min(attmap)) / (np.max(attmap)-np.min(attmap))
    segment = np.zeros((256, 256, 3))
    segment[:, :, 2] = attmap * 255.0
    segment[:, :, 1] = (1-attmap) * 255.0
    img = cv2.addWeighted(segment, 0.5, img, 0.5, 0)
    return img

def save_checkpoint(state, epoch, is_best, checkpoint='checkpoint', filename='checkpoint'):
    filepath = os.path.join(checkpoint, filename + '.pth.tar')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, filename+'_best.pth.tar'))

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
