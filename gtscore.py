import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.gtdata import DataAllocate, AudioVisualData
from model.detector import Framework, CAMCluster, ElementLoss_, CAMFeat_, em_cluster, CAMAudio
from utils.logger import AverageMeter, Logger
import sklearn.metrics

import cv2
import librosa
import numpy as np
import os
import pickle
from progress.bar import Bar
import argparse
import time
import shutil
import matplotlib.pyplot as plt
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser(description='AudioVisual Learning')

    parser.add_argument('-d', '--dataset', default='Flickr', type=str)
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')

    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=2, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--val-batch', default=1, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[50, 75],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--warm-up', dest='wp', default=100, type=int, 
                        help='warm up learning rate epoch')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--mix', default=4, type=int,
                        help='frames per batch for triplet loss')
    parser.add_argument('--rois', default=8, type=int,
                        help='rois per frame for visual stream')
    parser.add_argument('--size', default=3, type=int,
                        help='size for roi-align')
    parser.add_argument('--val-per-epoch','-tp', dest='tp', default=2, type=int,
                        help='number of training epoches between test (default: 30)')
    parser.add_argument('--iter-size', '-is', dest='its', default=4, type=int,
                        help='the forward-backward times within each iteration')

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

with open('utils/gtdata.pkl', 'rb') as f:
    info = pickle.load(f, encoding='bytes')

class Evaluator():

    def __init__(self):
        super(Evaluator, self).__init__()
        self.ciou = []

    def cal_CIOU(self, infer, gtmap, thres=0.01):
        gtmap = gtmap.reshape([256, 256])
        infer = infer.reshape([256, 256])
        infer_map = np.zeros((256, 256))
        infer_map[infer>=thres] = 1
        ciou = np.sum(infer_map*gtmap) / (np.sum(gtmap)+np.sum(infer_map*(gtmap==0)))
        self.ciou.append(ciou)
        return ciou

    def cal_AUC(self):
        results = []
        for i in range(21):
            result = np.sum(np.array(self.ciou)>=0.05*i)
            result = result / len(self.ciou)
            results.append(result)
        x = [0.05*i for i in range(21)]
        auc = sklearn.metrics.auc(x, results)
        print(results)
        return auc

    def final(self):
        ciou = np.mean(np.array(self.ciou)>=0.5)
        return ciou

    def clear(self):
        self.ciou = []

def main():

    f = open('utils/gtest.txt', 'r')
    pool = []
    while True:
        num = f.readline()
        if len(num) == 0:
            break
        pool.append(num.split('.')[0])

    start_epoch = args.start_epoch
    
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    
    print('Preparing Dataset %s' % args.dataset)
    
    valset = AudioVisualData(info, mix=args.mix, pool=pool, training=False)
    valloader = data.DataLoader(valset, batch_size=args.val_batch, shuffle=False, num_workers=args.workers, 
                                collate_fn=DataAllocate)
    
    framework = Framework(args.pre, args.mix, 1, args.rois, args.size)
    cluster = CAMCluster(framework)
    evaluator = Evaluator()
    camfeat = CAMAudio(framework)
    element = ElementLoss_(args.mix, 1)
    cluster_a = np.load('utils/cluster_a.npy')
    cluster_v = np.load('utils/cluster_v.npy')
    cluster_a = torch.FloatTensor(cluster_a.T)
    cluster_v = torch.FloatTensor(cluster_v)

    if use_gpu:
        framework = framework.cuda()
        cluster = cluster.cuda()
        camfeat = camfeat.cuda()
        element = element.cuda()
        cluster_a = cluster_a.cuda()
        cluster_v = cluster_v.cuda()
        
    args.checkpoint = os.path.join(args.checkpoint, args.dataset)
    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoint)
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        framework.load_state_dict(checkpoint['state_dict'], strict=True)
        start_epoch = checkpoint['epoch']
    
    if args.evaluate:
        ciou, auc = test_cam(valloader, framework, camfeat, element, evaluator, cluster_a, cluster_v, start_epoch, use_gpu)
        print(ciou, auc)
        return

def test_avg(dataloader, model, cam, element, evaluator, cluster_a, cluster_v, epoch, use_gpu):
    model.eval()
    model.audio.gru.train()

    data_time = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(dataloader))

    for batch_idx, (audio, visual, roi, gtmap, box) in enumerate(dataloader):
        audio = audio.view(args.val_batch * args.mix, *audio.shape[-2:])
        visual = visual.view(args.val_batch * args.mix, *visual.shape[-3:])
        roi = roi.view(args.val_batch, args.mix, args.rois, 4)
        gt = gtmap.numpy()

        data_time.update(time.time() - end)
        if use_gpu:
            audio = audio.cuda()
            visual = visual.cuda()
            roi = roi.cuda()

        data_time.update(time.time() - end)
        pred_a, feat_a, pred_v, feat_v = model(audio, visual, roi, cluster_a, cluster_v, False)
        feat_v = model.discrim.spa_conv(feat_v)
        feat_v = feat_v.permute([0, 2, 3, 1]).contiguous()

        feat_a = model.discrim.temp_conv(feat_a)
        feat_a = model.discrim.temp_pool(feat_a)

        feat_a = feat_a.view(args.val_batch, 1, 512)
        feat = torch.cat([feat_a.repeat(1, 256, 1),
                          feat_v.view(args.val_batch, 256, 512)], -1)
        feat = feat.view(-1, 1024)
        cams = model.discrim.auto_align(feat)
        cams = cams.view(args.val_batch, 256)
        cams = torch.softmax(-cams*10, -1)
        cams = cams / torch.max(cams, 1)[0].unsqueeze(-1)
        cams = torch.nn.functional.interpolate(cams.view(args.val_batch, 1, 16, 16), (256, 256),
                                               mode='bilinear')
        cams = cams.detach().cpu().numpy()
        for idx in range(args.val_batch):
            cam_visualize(batch_idx * args.val_batch + idx, visual[idx: idx + 1], cams[idx],
                          None, box[idx])
            ciou = evaluator.cal_CIOU(cams[idx], gt[idx], 0.1)
            end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | CIOU: {ciou:.3f}'.format(
            batch=batch_idx + 1,
            size=len(dataloader),
            data=data_time.val,
            ciou = ciou
        )
        bar.next()

    bar.finish()
    return evaluator.final(), evaluator.cal_AUC()

def test_cam(dataloader, model, cam, element, evaluator, cluster_a, cluster_v, epoch, use_gpu):
    model.eval()
    model.audio.gru.train()

    data_time = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(dataloader))

    for batch_idx, (audio, visual, roi, gtmap, box) in enumerate(dataloader):
        audio = audio.view(args.val_batch * args.mix, *audio.shape[-2:])
        visual = visual.view(args.val_batch * args.mix, *visual.shape[-3:])
        roi = roi.view(args.val_batch, args.mix, args.rois, 4)
        gt = gtmap.numpy()

        data_time.update(time.time() - end)
        if use_gpu:
            audio = audio.cuda()
            visual = visual.cuda()
            roi = roi.cuda()

        data_time.update(time.time() - end)
        pred_a, feat_a, pred_v, feat_v = model(audio, visual, roi, cluster_a, cluster_v, False)
        feat_v = model.discrim.spa_conv(feat_v)
        feat_v = feat_v.permute([0, 2, 3, 1]).contiguous()

        feat_a, ratio, coeff = cam(pred_a, feat_a)
        feat_a = model.discrim.temp_conv(feat_a)
        feat_a = model.avalign.temp_pool(feat_a)

        # coeff = torch.nn.functional.adaptive_max_pool2d(coeff, (feat_a.shape[-2], feat_a.shape[-1]))
        # coeff = coeff.view(*coeff.shape[:2], -1)
        # feat_a = feat_a.view(*feat_a.shape[:2], -1)
        # feat_a = torch.sum(feat_a*coeff, -1) / torch.sum(coeff+1e-10, -1)

        feat_a = feat_a.view(args.val_batch, 7, 1, 512)
        feat = torch.cat([feat_a.repeat(1, 1, 256, 1),
                          feat_v.view(args.val_batch, 1, 256, 512).repeat([1, 7, 1, 1])], -1)
        feat = feat.view(-1, 1024)
        cams = model.discrim.auto_align(feat)
        cams = cams.view(args.val_batch, 7, 256)
        cams = torch.softmax(-cams*10, -1)
        cams = cams / torch.max(cams, -1)[0].unsqueeze(-1)
        cams = torch.nn.functional.interpolate(cams.view(args.val_batch, 7, 16, 16), (256, 256),
                                               mode='bilinear')
        cams = cams.detach().cpu().numpy()
        for idx in range(args.val_batch):
            cam_visualize(batch_idx*args.val_batch+idx, visual[idx: idx+1], cams[idx], None, box[idx])
            index = (ratio[idx].cpu().numpy()>0)
            pred = pred_a[idx].detach().cpu().numpy()
            index[np.argmax(ratio[idx].cpu().numpy())] = 1
            camp = np.sum(cams[idx][index]*pred.reshape(7, 1, 1)[index], 0) / np.sum(pred[index])
            # camp = np.max(cams[idx][index], 0)
            # cam_visualize(batch_idx * args.val_batch + idx, visual[idx: idx + 1], np.expand_dims(camp, 0),
            #               None, box[idx])
            ciou = evaluator.cal_CIOU(camp, gt[idx], 0.1)
            end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | CIOU: {ciou:.3f}'.format(
            batch=batch_idx + 1,
            size=len(dataloader),
            data=data_time.val,
            ciou = ciou
        )
        bar.next()

    bar.finish()
    return evaluator.final(), evaluator.cal_AUC()

def gt_visualize(batch_idx, visual, gtmap, boxes=None):
    assert visual.shape[0] == gtmap.shape[0]
    segments = visual.shape[0]
    visual = visual.cpu().numpy()
    visual = visual.transpose([0, 2, 3, 1])
    visual = visual * np.array([0.229, 0.224, 0.225])
    visual = visual + np.array([0.485, 0.456, 0.406])
    visual = np.clip(visual, 0.0, 1.0)
    for seg in range(segments):
        actmap = gtmap[seg]
        actmap = np.expand_dims(actmap, -1)
        img = visual[seg]
        if boxes is not None:
            for box in boxes:
                (xmin, ymin, xmax, ymax) = box
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        plt.imsave('./gtvis/'+str(batch_idx)+'_'+str(seg)+'gt.jpg', 0.6*actmap+0.4*img)

def cam_visualize(batch_idx, visual, cam, index=None, boxes=None, thres=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    segments = cam.shape[0]
    visual = visual.cpu().numpy()
    visual = visual.transpose([0, 2, 3, 1])
    visual = visual * np.array([0.229, 0.224, 0.225])
    visual = visual + np.array([0.485, 0.456, 0.406])
    visual = np.clip(visual, 0.0, 1.0)
    for seg in range(segments):
        actmap = cam[seg]
        if thres is not None:
            actmap[actmap>thres] = 1
        actmap = actmap * 255
        actmap = actmap.astype(np.uint8)
        actmap = cv2.applyColorMap(actmap, cv2.COLORMAP_JET)[:, :, ::-1]
        actmap = actmap / 255.0
        img = visual[0]
        if index is not None:
            img = cv2.putText(img, str(index), (0, 40), font, 1.2, (255, 255, 255), 2)
        # if boxes is not None:
        #     for box in boxes:
        #         (xmin, ymin, xmax, ymax) = box
        #         cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        plt.imsave('./gtvis/'+str(batch_idx)+'_'+str(seg)+'.jpg', 0.4*actmap+0.6*img)

if __name__ == '__main__':
    main()
