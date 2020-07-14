import torch
from dataloader import AudioVisualData, DataAllocate
from testdata import TestData, TestAllocate
from base_model import resnet18
from stage_one import MTask
from stage_two import Align, filter_prob, contrastive, Location
import torch.utils.data as data
import argparse
from progress.bar import Bar
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def train_stage_one(model, trainloader, optimizer):
    losses = []
    bar = Bar('Processing', max=len(trainloader))
    bceloss = torch.nn.BCELoss(size_average=True, reduce=True)
    celoss = torch.nn.CrossEntropyLoss(size_average=False, reduce=False)
    for batch_idx, (spec, img, label_a, label_v) in enumerate(trainloader):
        spec = spec.cuda()
        img = img.cuda()
        label_a = label_a.cuda()
		label_v = label_v.cuda()
        avc, cls_a, cls_v = model(spec, img)
        gt_avc = torch.eye(avc.shape[0]).type(torch.LongTensor).view(-1).to(avc.device)
        avcloss = celoss(avc.view(-1, 2), gt_avc)
        avcloss = torch.mean(avcloss[gt_avc==1])/2 + torch.mean(avcloss[gt_avc==0])/2
        aloss = bceloss(torch.sigmoid(cls_a), label_a)
        vloss = bceloss(torch.sigmoid(cls_v), label_v)
        loss = aloss + vloss + avcloss
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        bar.suffix = '({batch}/{size}) ALoss: {aloss:.3f}| VLoss: {vloss:.3f}| AVCLoss: {avcloss:.3f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            aloss=aloss.item(),
            vloss=vloss.item(),
            avcloss=avcloss.item()
        )
        bar.next()
    bar.finish()
    return sum(losses)/len(losses)

def train_stage_two(model, trainloader, optimizer):
    losses = []
    bar = Bar('Processing', max=len(trainloader))
    bceloss = torch.nn.BCELoss(size_average=True, reduce=True)
    celoss = torch.nn.CrossEntropyLoss(size_average=False, reduce=False)
    for batch_idx, (spec, img, label_a, label_v) in enumerate(trainloader):
        spec = spec.cuda()
        img = img.cuda()
        label_a = label_a.cuda()
		label_v = label_v.cuda()
        avc, cls_a, cls_v, fine_a, fine_v = model(spec, img)
        gt_avc = torch.eye(avc.shape[0]).type(torch.LongTensor).view(-1).to(avc.device)
        avcloss = celoss(avc.view(-1, 2), gt_avc)
        avcloss = torch.mean(avcloss[gt_avc==1])/2 + torch.mean(avcloss[gt_avc==0])/2
        aloss = bceloss(torch.sigmoid(cls_a), label_a)
        vloss = bceloss(torch.sigmoid(cls_v), label_v)
        
        eff = filter_prob(cls_a, cls_v, 0.3)
        distance = 2-2*contrastive(fine_a, fine_v)
        pos = torch.eye(distance.shape[0]).to(fine_a.device)
        neg = 1 - pos
        pos_dis = torch.sum(distance*eff*pos) / torch.sum(pos*eff+1e-10)
        neg_dis = torch.sum(distance*eff*neg) / torch.sum(neg*eff+1e-10)
        dloss = pos_dis + torch.nn.functional.relu(1.4-neg_dis.sqrt()).pow(2)
        
        loss = aloss + vloss + avcloss + dloss
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        bar.suffix = '({batch}/{size}) ALoss: {aloss:.3f}| VLoss: {vloss:.3f}| AVCLoss: {avcloss:.3f}| DLoss: {dloss:.3f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            aloss=aloss.item(),
            vloss=vloss.item(),
            avcloss=avcloss.item(),
            dloss=dloss.item()
        )
        bar.next()
    bar.finish()
    return sum(losses)/len(losses)

def test(model, testloader):
    cious = []
    model.eval()
    locations = []
    gtmaps = []
    aln_as = []
    aln_vs = []
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (spec, img, label, box) in enumerate(testloader):
        spec = spec.cuda()
        img = img.cuda()
        avc, cls_a, cls_v, fine_a, fine_v, cam_a, cam_v, aln_a, aln_v = model(spec, img)
        cls_a = torch.nn.functional.sigmoid(cls_a)
        similarity = torch.einsum('bij,bik->bjk', fine_a, fine_v)
        similarity = torch.nn.functional.softmax(similarity/0.1, -1)
        similarity = similarity.view(*similarity.shape[:2], 16, 16)
        similarity = torch.nn.functional.interpolate(similarity, (256, 256), 
                                                     mode='bilinear')
        cam_a = torch.nn.functional.interpolate(cam_a, (256, 256), 
                                                     mode='bilinear')
        cam_v = torch.nn.functional.interpolate(cam_v, (256, 256), 
                                                     mode='bilinear')
        location = visualize(img, similarity, torch.sigmoid(cls_a)[0], batch_idx)
        locations.append(location)
        gtmaps.append(label[0].numpy())
        ciou = cal_ciou(location, label, 0.01)
        cious.append(ciou)
        aln_as.append(aln_a.cpu().detach().numpy())
        aln_vs.append(aln_v.cpu().detach().numpy())
        bar.suffix = '({batch}/{size}) CIOU: {ciou:.3f}'.format(
            batch=batch_idx + 1,
            size=len(testloader),
            ciou=ciou
        )
        bar.next()
    bar.finish()
    print(np.sum(np.array(cious)>=0.5)/len(cious))
    np.save('location', np.array(locations))
    np.save('gtmap', np.array(gtmaps))
    np.save('aln_as', np.array(aln_as))
    np.save('aln_vs', np.array(aln_vs))
        
def cal_ciou(location, gtmap, thres):
    gtmap = gtmap[0].numpy()
    assert location.shape == gtmap.shape
    ciou = np.sum(gtmap[location>thres]) / (np.sum(gtmap)+np.sum(gtmap[location>thres]==0))
    return ciou

def visualize(img, location, prob, idx):
    prob = prob.cpu().detach().numpy()
    img = img.permute(0, 2, 3, 1)
    img = img.cpu().numpy()[0]
    location = torch.nn.functional.relu(location).cpu().detach().numpy()[0]
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    location = np.sum(location*prob.reshape(7, 1, 1), 0) / np.sum(prob)
    location = location / np.max(location)
    # plt.imsave('vis/%d.jpg'%idx, 0.5*img+0.5*np.expand_dims(location, -1))
    # plt.imsave('vis/%d.jpg'%idx, img)
    return location

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='AudioVisual')
    parser.add_argument('--train_batch', type=int, default=16)
    parser.add_argument('--val_batch', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pretrained', type=int, default=0)
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--evaluate', type=int, default=0)
    args = parser.parse_args()

    trainset = AudioVisualData()
    testset = TestData()
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, 
                                  collate_fn=DataAllocate, num_workers=args.num_workers)
    testloader = data.DataLoader(testset, batch_size=args.val_batch, shuffle=False, 
                                  collate_fn=TestAllocate, num_workers=args.num_workers)
    
    vision_net = resnet18(modal='vision', pretrained=True)
    audio_net = resnet18(modal='audio')
    
    if args.evaluate:
        net = Location(vision_net, audio_net).cuda()
        net.load_state_dict(torch.load(args.path))
        test(net, testloader)
        exit()
    
    net = MTask(vision_net, audio_net).cuda() if args.stage == 1 \
        else Align(vision_net, audio_net).cuda()
    if args.pretrained:
        net.load_state_dict(torch.load(args.path), strict=False)
    
    params = list(net.parameters())
    optimizer = torch.optim.SGD(params=params, lr=args.learning_rate, 
                                momentum=args.momentum, 
                                weight_decay=args.weight_decay)
    
    for e in range(args.epoch):
        loss = train_stage_one(net, trainloader, optimizer) if args.stage == 1 \
            else train_stage_two(net, trainloader, optimizer)
        torch.save(net.state_dict(), 'ckpt/20k/stage_%d_%d_%.3f.pth'%(args.stage, e, loss))