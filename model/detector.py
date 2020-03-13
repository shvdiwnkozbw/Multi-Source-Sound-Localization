import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision
import numpy as np

from unet.unet import Unet
from resnet.resnet import resnet18
from vggish.vggish import vggish
from gcn import gcn
from resnet.resnet import resnet50
from mobile_crnn.models import mobilecrnn_v2
from mobile_crnn.grad_cam import FeatureExtractor, ModelOutputs

def normalize(feat):
    return

def GraphPooling(x):
    num_node = x.shape[1]
    x = x.permute(0, 2, 1).contiguous()
    x = torch.sum(x, -1) / num_node
    return x

def l2norm(x, dim=-1):
    norm = torch.pow(x, 2).sum(dim=dim, keepdim=True).sqrt()
    x_norm = torch.div(x, norm)
    return x_norm, norm

def em_cluster(embed, centers):
    # embed(batchsize, channel, points)
    # centers(batchsize)
    if centers == 1:
        cluster = torch.ones_like(embed[:, :1, :])
        return cluster
    embed = embed.permute(0, 2, 1).contiguous()
    embed = embed.view(embed.shape[0], -1, embed.shape[-1])
    randidx = torch.randperm(embed.shape[1])[:centers]
    pos = [embed[:, randidx[idx], :].unsqueeze(1) for idx in range(centers)]
    pos = torch.stack(pos, 1)
    for _ in range(10):
        dist = embed.unsqueeze(1) - pos
        dist = torch.sum(torch.pow(dist, 2), -1).sqrt()
        # cluster = torch.argmin(dist, 1)
        cluster = torch.softmax(-1000*dist, 1)
        for idx in range(centers):
            # vec = embed * (cluster==idx).type(torch.cuda.FloatTensor).unsqueeze(-1)
            # vec = torch.sum(vec, 1, keepdim=True) / torch.sum(cluster==idx, 1, keepdim=True).type(torch.cuda.FloatTensor)
            vec = embed * cluster[:, idx, :].unsqueeze(-1)
            vec = torch.sum(vec, 1) / torch.sum(cluster[:, idx, :], 1).unsqueeze(1)
            pos[:, idx, :, :] = vec.unsqueeze(1)
    return cluster.detach()

def cal_IOU(rois):
    batchsize, num_frame, num_rois = rois.shape[:3]
    x_min, y_min, x_max, y_max = rois.permute(3, 0, 1, 2).contiguous()
    min_x_min = torch.min(x_min.unsqueeze(2), x_min.unsqueeze(3))
    max_x_min = torch.max(x_min.unsqueeze(2), x_min.unsqueeze(3))
    min_x_max = torch.min(x_max.unsqueeze(2), x_max.unsqueeze(3))
    max_x_max = torch.max(x_max.unsqueeze(2), x_max.unsqueeze(3))
    min_y_min = torch.min(y_min.unsqueeze(2), y_min.unsqueeze(3))
    max_y_min = torch.max(y_min.unsqueeze(2), y_min.unsqueeze(3))
    min_y_max = torch.min(y_max.unsqueeze(2), y_max.unsqueeze(3))
    max_y_max = torch.max(y_max.unsqueeze(2), y_max.unsqueeze(3))
    inter = torch.max(min_x_max-max_x_min, torch.zeros_like(min_x_max)) * torch.max(min_y_max-max_y_min, torch.zeros_like(min_y_max))
    union = (max_x_max-min_x_min) * (max_y_max-min_y_min)
    weight = inter / union
    return weight

class StatFreq(nn.Module):

    def __init__(self, class_a, class_v):
        super(StatFreq, self).__init__()
        self.class_a = class_a
        self.class_v = class_v
        self.stat_a = torch.zeros(self.class_a)
        self.stat_v = torch.zeros(self.class_v)
        self.co_a = torch.zeros(self.class_a, self.class_a)
        self.co_v = torch.zeros(self.class_v, self.class_v)
        self.co_av = torch.zeros(self.class_a, self.class_v)

    def process_v(self, prob):
        _, ids = torch.sort(prob, 1, descending=True)
        batchidx = torch.arange(prob.shape[0]).unsqueeze(1).repeat([1, self.class_v])
        levelidx = torch.arange(1, 1+self.class_v)
        level = torch.pow(0.95, levelidx.type(torch.cuda.FloatTensor)).unsqueeze(0)
        level = level.repeat([prob.shape[0], 1])
        prob = prob / torch.max(prob, 1)[0].unsqueeze(1)
        rank = torch.zeros_like(prob)
        rank[batchidx, ids] = level
        prob = prob * rank
        return prob

    def pull_result(self):
        return self.stat_a, self.stat_v, self.co_a, self.co_v, self.co_av

    def process_label(self, label_a, label_t, label_v, label_r):
        assert label_v.shape[0] == label_a.shape[0]
        segments = label_a.shape[0]
        for seg in range(segments):
            thres = torch.min(torch.tensor(0.8).cuda(), torch.max(label_a[seg]))
            index_a = torch.arange(self.class_a)[label_a[seg]>=thres][:5]
            thres = torch.min(torch.tensor(0.5).cuda(), torch.max(label_v[seg]))
            index_v = torch.arange(self.class_v)[label_v[seg]>=thres][:5]
            self.stat_a[index_a] += 1
            self.stat_v[index_v] += 1
            self.co_a[index_a.unsqueeze(0), index_a.unsqueeze(1)] += 1
            self.co_v[index_v.unsqueeze(0), index_v.unsqueeze(1)] += 1
            self.co_av[index_a.unsqueeze(0), index_v.unsqueeze(1)] += 1

    def update(self, index_a, index_v):
        self.stat_a[index_a] += 1
        self.stat_v[index_v] += 1
        self.co_a[index_a.unsqueeze(0), index_a.unsqueeze(1)] += 1
        self.co_v[index_v.unsqueeze(0), index_v.unsqueeze(1)] += 1
        self.co_av[index_a.unsqueeze(0), index_v.unsqueeze(1)] += 1

    def process_prob(self, label_a, label_t, label_v, label_r):
        assert label_t.shape[0] == label_v.shape[0]
        assert label_r.shape[0] / label_v.shape[0] == 8
        label_v = self.process_v(label_v)
        label_r = self.process_v(label_r)
        segments = label_v.shape[0]
        rois = label_r.shape[0] // segments
        label_r = label_r.view(segments, rois, self.class_v)
        for seg in range(segments):
            thres = torch.min(torch.tensor(0.4).cuda(), torch.max(label_t[seg]))
            index_a = torch.arange(self.class_a)[label_t[seg]>=thres][:5]
            label = torch.max(label_r[seg], 0)[0]
            index_v = torch.arange(self.class_v)[label>=0.5][:10]
            self.update(index_a, index_v)
        thres = torch.min(torch.tensor(0.4).cuda(), torch.max(label_a))
        index_a = torch.arange(self.class_a)[label_a[0]>=thres][:5]
        index_v = torch.arange(self.class_v)[torch.max(label_v, 0)[0]>=0.5][:10]
        self.update(index_a, index_v)

    def forward(self, label_a, label_t, label_v, label_r):
        self.process_prob(label_a, label_t, label_v, label_r)

class GradCam:
    def __init__(self, model, target_layer_names):
        self.model = model
        self.model.eval()
        self.model.gru.train()
        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        features, output, temps = self.extractor(input)
        embed = []
        for index in range(output.shape[-1]):
            one_hot = torch.zeros_like(output)
            one_hot[:, index] = 1
            one_hot = one_hot * output
            one_hot = torch.sum(one_hot)

            self.model.features.zero_grad()
            self.model.gru.zero_grad()
            self.model.temp_pool.zero_grad()
            self.model.outputlayer.zero_grad()
            one_hot.backward(retain_graph=True)

            grads_val = self.extractor.get_gradients()[-1]
            target = features[-1]

            weights = torch.mean(grads_val, dim=(2, 3))
            cam = torch.sum(weights.view(weights.shape[0], weights.shape[1], 1, 1)*target.unsqueeze(1), 1)
            cam = torch.relu(cam)
            norm = torch.max(cam.view(weights.shape[0], -1), 1)[0]
            cam = cam / norm.view(norm.shape[0], 1, 1)
            feature = target * cam.unsqueeze(1)
            feature = feature.transpose(1, 2).contiguous().flatten(-2)
            feature, _ = self.model.gru(feature)
            embed.append(feature)

        return torch.stack(embed, 1), temps

class Extractor(nn.Module):

    def __init__(self, pretrained, mix, frames, rois, size):
        super(Extractor, self).__init__()
        self.scale = 16.0
        self.mix = mix
        self.frames = frames
        self.rois = rois
        self.visual = resnet18(pretrained=pretrained, num_classes=7, additional=True)
        self.audio = mobilecrnn_v2(outputdim=7, pretrained=pretrained)
        self.align = RoIAlignAvg(size, size, 1.0/self.scale)
        self.discrim = DiscrimHead(self.mix, self.frames, self.visual.layer5, 512)
        self.temp_pool = nn.AdaptiveAvgPool1d((self.frames))
        self.grad_cam = GradCam(model=self.audio, target_layer_names=['11'])
        self.size = size

    def forward(self, audio, visual, rois):
        feat = self.visual(visual, 'bottom')
        rois = self.align(feat, rois)
        embed_v, prob_v = self.visual(rois, 'comp')
        embed_v = embed_v.view(-1, self.frames, self.rois, embed_v.shape[-1])
        prob_v = prob_v.view(-1, self.frames, self.rois, prob_v.shape[-1])
        prob_v = torch.softmax(prob_v, -1)
        embed_a, prob_a = self.grad_cam(audio)
        embed_a = self.temp_pool(embed_a.permute(0, 2, 1).contiguous())
        embed_a = embed_a.permute(0, 2, 1).contiguous()
        return embed_a, prob_a, embed_v, prob_v

class CAMCluster(nn.Module):

    def __init__(self, model):
        super(CAMCluster, self).__init__()
        self.model = model
        self.model.eval()

    def forward(self, audio, visual):
        cams = []
        feat_a = self.model.audio(audio, 'embed')
        feat_v = self.model.visual(visual, 'bottom')
        common, differ, feat_a, feat_v = self.model.discrim(feat_a, feat_v, True)
        for idx in range(2):
            one_hot = torch.zeros_like(common.view(-1, 2))
            one_hot[:, idx] = 1
            one_hot = torch.sum(one_hot*common.view(-1, 2))
            self.model.zero_grad()
            one_hot.backward(retain_graph=True)
            grad = self.model.discrim.gradient[-1]
            weights = torch.sum(grad.view(grad.shape[0], grad.shape[1], -1), -1)
            cam = torch.sum(weights.view(weights.shape[0], weights.shape[1], 1, 1)*feat_v, 1)
            cams.append(cam)
        # cam = torch.relu(cam)
        # norm = torch.max(cam.view(cam.shape[0], -1), 1)[0]
        # cam = cam / (norm.view(norm.shape[0], 1, 1)+1e-10)
        cam = torch.clamp(cam, 0.0, 1.0)
        return [common, differ], cam

class CAMFeat(nn.Module):

    def __init__(self, model):
        super(CAMFeat, self).__init__()
        self.model = model

    def cam_feat(self, grad, feat):
        weight = torch.mean(grad.contiguous().view(grad.shape[0], grad.shape[1], -1), -1)
        cam = torch.sum(weight.view(weight.shape[0], weight.shape[1], 1, 1) * feat, 1)
        return cam.view(cam.shape[0], feat.shape[2], feat.shape[3])

    def cal_cluster(self, weight, centers, feat):
        cluster = em_cluster(weight.view(*weight.shape[:2], -1), centers)
        feat = feat.view(*feat.shape[:2], -1)
        feat = feat.unsqueeze(1) * cluster.unsqueeze(2)
        feat = torch.sum(feat, -1) / torch.sum(cluster, -1, keepdim=True)
        return feat, cluster

    def forward(self, pred_a, pred_v, feat_a, feat_v):
        weights = []
        class_a = pred_a.shape[-1]
        class_v = pred_v.shape[-1]
        for seg in range(class_a):
            one_hot = torch.zeros_like(pred_a)
            one_hot[:, seg] = 1
            one_hot = torch.sum(one_hot * pred_a)
            self.model.zero_grad()
            one_hot.backward(retain_graph=True)
        for seg in range(class_v):
            one_hot = torch.zeros_like(pred_v)
            one_hot[:, seg] = 1
            one_hot = torch.sum(one_hot * pred_v)
            self.model.zero_grad()
            one_hot.backward(retain_graph=True)
        self.model.zero_grad()
        grads = self.model.gradient
        for seg, grad in enumerate(grads):
            feat = feat_a if seg < class_a else feat_v
            cam = self.cam_feat(grad, feat)
            weights.append(cam.unsqueeze(1))
        weight_a = torch.stack(weights[:class_a], 1)
        weight_v = torch.stack(weights[class_a:], 1)
        cluster_a = self.cal_cluster(weight_a, 1, feat_a)
        cluster_v = self.cal_cluster(weight_v, 2, feat_v)
        return [cluster_a[0], cluster_v[0]], [cluster_a[1], cluster_v[1]]

class CAMVisual(nn.Module):

    def __init__(self, model):
        super(CAMVisual, self).__init__()
        self.model = model

    def cam_feat(self, feat):
        weight = self.model.visual.fc.weight
        feat = feat.view(*feat.shape[:2], -1)
        cam = torch.matmul(weight.unsqueeze(0), feat)
        cam = torch.relu(cam - self.thres)
        return cam.detach()

    def forward(self, cam_feat, feat, thres=0.1):
        self.thres = thres
        cam = self.cam_feat(cam_feat)
        cam = cam.view(*cam.shape[:2], 1, *feat.shape[-2:])
        return cam
#        feat = feat.unsqueeze(1) * cam
#        return feat.view(-1, *feat.shape[-3:])

class CAMAudio(nn.Module):

    def __init__(self, model):
        super(CAMAudio, self).__init__()
        self.model = model

    def cam_feat(self, grad, feat):
        weight = torch.sum(grad.contiguous().view(grad.shape[0], grad.shape[1], -1), -1)
        cam = torch.sum(weight.view(weight.shape[0], weight.shape[1], 1, 1) * feat, 1)
        pcam = torch.relu(cam-0.1)
        pcam[pcam>0] = 1
        return pcam.view(pcam.shape[0], 1, -1).detach(), \
               torch.sum(pcam.view(pcam.shape[0], -1), -1) / (pcam.shape[1]*pcam.shape[2]), \
               cam.unsqueeze(1)

    def forward(self, pred_a, feat_a):
        feats = []
        ratios = []
        coeffs = []
        class_a = pred_a.shape[-1]
        for seg in range(class_a):
            one_hot = torch.zeros_like(pred_a)
            one_hot[:, seg] = 1
            one_hot = torch.sum(one_hot * pred_a)
            self.model.zero_grad()
            one_hot.backward(retain_graph=True)
        self.model.zero_grad()
        grads = self.model.gradient
        for seg, grad in enumerate(grads):
            cam, ratio, coeff = self.cam_feat(grad, feat_a)
            # feat = torch.sum(cam*feat_a.view(*feat_a.shape[:2], -1), -1) / torch.sum(cam, -1)
            feat = feat_a * cam.view(feat_a.shape[0], 1, *feat_a.shape[-2:])
            feats.append(feat)
            ratios.append(ratio.detach())
            coeffs.append(coeff.detach())
        feats = torch.stack(feats, 1)
        feats = feats.view(-1, *feats.shape[-3:])
        ratios = torch.stack(ratios, 1)
        coeffs = torch.stack(coeffs, 1)
        return feats, ratios, coeffs.view(-1, *coeffs.shape[-3:])

class AVAlign(nn.Module):

    def __init__(self, frame, roi, model):
        super(AVAlign, self).__init__()
        self.frame = frame
        self.roi = roi
        self.model = model
        self.temp_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.spa_pool = nn.AdaptiveMaxPool2d((1, 1))

    def cam_pool(self, cam, feat):
        fg = feat.unsqueeze(1) * cam
        fg = torch.sum(fg.view(*fg.shape[:3], -1), -1) / \
               torch.sum(cam.view(*cam.shape[:3], -1)+1e-10, -1)
        bg = (torch.max(cam, 1)[0] == 0).type(torch.cuda.FloatTensor)
        bg = torch.sum(feat.view(*feat.shape[:2], -1)*bg.view(*bg.shape[:2], -1), -1) /\
             torch.sum(bg.view(*bg.shape[:2], -1)+1e-10, -1)
        return fg, bg

    def forward(self, feat_a, pred_a, feat_v, pred_v, cam):
        # feat_a(mix, class, channel)
        # feat_v(mix, frame, channel)
        # pred_a(mix, class)
        # pred_v(mix, frame, class)
        pred_v = pred_v.view(-1, self.frame, pred_v.shape[-1])
        feat_a = self.model.temp_conv(feat_a)
        feat_a = self.temp_pool(feat_a)
        feat_a = feat_a.view(-1, pred_a.shape[-1], feat_a.shape[1])
        feat_v = self.model.spa_conv(feat_v)
        feat_v = self.cam_pool(cam, feat_v)
        feat_v = feat_v.view(-1, self.frame, pred_v.shape[-1], feat_v.shape[-1])

        segments = feat_a.shape[0]
        same = [None] * segments
        differ = [None] * segments
        embed_co = []
        embed_di = []
        pred_v = torch.sigmoid(pred_v)
        index_v = pred_v.view(segments, self.frame, pred_v.shape[-1])
        index_a = (pred_a > 0.3)
        index_a[torch.arange(segments), torch.argmax(index_a, 1)][torch.max(index_a, 1)[0]>0.2] = 1
        for seg in range(segments):
            same[seg] = []
            differ[seg] = []
            rank = seg + 1 if seg % 2 == 0 else seg - 1
            if torch.sum(index_a[seg]>0) == 0:
                continue
            eff_a = torch.arange(index_a.shape[-1])[index_a[seg]>0]
            for index in eff_a:
                num = int(pred_a[seg, index]*self.frame)
                eff_v = torch.arange(index_v.shape[1])[index_v[seg, :, index]>0.3]
                differ[seg].append(torch.cat([feat_a[seg, [index]*num], feat_v[rank, torch.randperm(self.frame)[:num], torch.randint(0, pred_v.shape[-1], (num,))]], -1))
                if len(eff_v):
                    same[seg].append(torch.cat([feat_a[seg, [index]*len(eff_v)], feat_v[seg, eff_v, index]], -1))

        for ele in same:
            if len(ele):
                embed_co.append(torch.cat(ele, 0))
        for ele in differ:
            if len(ele):
                embed_di.append(torch.cat(ele, 0))
        try:
            embed_co = torch.cat(embed_co, 0)
            embed_di = torch.cat(embed_di, 0)
            embed_co = F.mse_loss(self.model.transform_a(embed_co[:, :512]),
                                  self.model.transform_v(embed_co[:, 512:]), reduce=False)
            embed_co = embed_co.mean(1)
            embed_di = F.mse_loss(self.model.transform_a(embed_di[:, :512]),
                                  self.model.transform_v(embed_di[:, 512:]), reduce=False)
            embed_di = embed_di.mean(1)
            return embed_co, embed_di
        except:
            return None, None

class AlignLoss(nn.Module):

    def __init__(self, margin):
        super(AlignLoss, self).__init__()
        self.margin = margin
        self.celoss = nn.CrossEntropyLoss(size_average=False, reduce=False)

    def contrastive_loss(self, common, differ):
        if common is None:
            return torch.tensor(0.0).cuda()
        closs = torch.mean(common)
        dloss = torch.mean(torch.relu(self.margin-differ))
        loss = closs + dloss
        return loss

    def cross_entropy_loss(self, common, differ):
        if common is None:
            return torch.tensor(0.0).cuda()
        gt_co = torch.ones_like(common[:, 1]).type(torch.cuda.LongTensor)
        gt_di = torch.zeros_like(differ[:, 0]).type(torch.cuda.LongTensor)
        closs = self.celoss(common, gt_co)
        dloss = self.celoss(differ, gt_di)
        loss = torch.sum(closs) / float(common.shape[0]) + torch.sum(dloss) / float(differ.shape[0])
        return loss

    def forward(self, common, differ):
        loss = self.contrastive_loss(common, differ)
        return loss

class Framework(nn.Module):
    
    def __init__(self, pretrained, mix, frames, rois, size):
        super(Framework, self).__init__()
        self.scale = 16.0
        self.mix = mix
        self.frames = frames
        self.rois = rois
        self.size = (int(size), int(size))
        self.prior_v = resnet18(pretrained)
        self.prior_a = mobilecrnn_v2(pretrained=pretrained)
        self.visual = resnet18(pretrained, num_classes=7, additional=True)
        self.audio = mobilecrnn_v2(outputdim=7, pretrained=pretrained)
        self.roi_align = torchvision.ops.RoIAlign(self.size, 1.0/self.scale, -1)
        self.discrim = DiscrimHead(self.mix, self.frames, self.visual.layer5, 512)
        self.temp_pool = nn.AdaptiveMaxPool1d((self.frames))
        self.avalign = AVAlign(frames, rois, self.discrim)

    def align(self, feat, rois):
        batchsize, frames = rois.shape[:2]
        rois = rois.view(-1, 4)
        rois = torch.cat([torch.zeros_like(rois[:, :1]), rois], 1)
        for idx in range(batchsize*frames):
            rois[idx*self.rois: (idx+1)*self.rois, 0] = idx
        rois = self.roi_align(feat, rois)
        rois = rois.view(batchsize, frames*self.rois, *rois.shape[1:])
        return rois

    def linear_pool(self, feat):
        stride = feat.shape[-1] / self.frames
        point = np.arange(self.frames) * stride
        stride = int(np.ceil(stride))
        feat = torch.clamp(feat, 1e-7, 1.0)
        filt = [None] * self.frames
        for seg in range(self.frames):
            segment = feat[:, :, int(point[seg]): int(point[seg])+stride]
            filt[seg] = torch.sum(torch.pow(segment, 2), -1) / torch.sum(segment, -1)
        feat = torch.stack(filt, -1)
        return feat

    def spatial_pooling(self, feat):
#        feat N, C, H, W
        feat = torch.sum(feat, -1)
        feat = torch.sum(feat, -1)
        return feat

    def process_a(self, prob, cluster_a):
        label = prob.unsqueeze(-1) * cluster_a.unsqueeze(0)
        label = torch.mean(torch.topk(label, 1, 1)[0], 1)
        label = torch.min(2*label, torch.ones_like(label))
        return label

    def process_v(self, prob, cluster_v, roi=False):
        _, ids = torch.sort(prob, 1, descending=True)
        batchidx = torch.arange(prob.shape[0]).unsqueeze(1).repeat([1, 1000])
        if roi:
            levelidx = torch.arange(0, 1000)
            level = torch.pow(0.9, levelidx.type(torch.cuda.FloatTensor)).unsqueeze(0)
        else:
            levelidx = torch.arange(1, 1001)
            level = torch.pow(0.95, levelidx.type(torch.cuda.FloatTensor)).unsqueeze(0)
        level = level.repeat([prob.shape[0], 1])
        prob = prob / torch.max(prob, 1)[0].unsqueeze(1)
        rank = torch.zeros_like(prob)
        rank[batchidx, ids] = level
        prob = prob * rank
        label = prob.unsqueeze(-1) * cluster_v.unsqueeze(0)
        label = torch.mean(torch.topk(label, 2, 1)[0], 1)
        return label

    def generate_label(self, prob_a, prob_t, prob_v, prob_r, cluster_a, cluster_v):
#        prob_a(batchsize*mix, 527)
#        prob_t(batchsize*mix*frame, 527)
#        prob_v(batchsize*mix*frame, 1000)
#        prob_roi(batchsize*mix*frame*rois, 1000)
        label_a = self.process_a(prob_a, cluster_a)
        label_t = self.process_a(prob_t, cluster_a)
        label_v = self.process_v(prob_v, cluster_v)
        label_r = self.process_v(prob_r, cluster_v, True)
        label_r = torch.argmax(label_r, -1)
        return label_a.detach(), label_v.detach(), label_t.detach(), label_r.detach()
    
    def calculate_similarity(self, label_a, label_v):
        norm_a = torch.sqrt(torch.sum(torch.pow(label_a, 2), 1, keepdim=True))
        norm_v = torch.sqrt(torch.sum(torch.pow(label_v, 2), 1, keepdim=True))
        norm = torch.mm(norm_a, norm_v.permute(1, 0).contiguous())
        inter = torch.mm(label_a, label_v.permute(1, 0).contiguous())
        matrix = inter / norm
        common = torch.diag(matrix)
        common = common.view(-1, self.frames)
        segments = matrix.shape[0] / self.frames
        for seg in range(segments-1):
            gallery = matrix[seg*self.mix: (seg+1)*self.mix, (seg+1)*self.mix:]
        return matrix

    def sample_prior(self, label_a, label_v):
#        label_a(batchsize*mix, frame, cluster)
#        label_v(batchsize*mix, frame, cluster)
        label_a = label_a.view(-1, self.frames, label_a.shape[-1])
        label_v = label_v.view(-1, self.frames, label_v.shape[-1])
        max_value = torch.max(label_a, -1)[0]
        thres = torch.max(max_value, 1)[0].unsqueeze(1)
        thres = torch.min(thres, 0.8*torch.ones_like(thres))
        if self.frames == 1:
            thres = torch.min(torch.max(thres), torch.tensor(0.6).cuda())
            thres = thres * torch.ones_like(max_value)
        mask = (max_value>=thres)
        mask = mask.type(torch.cuda.FloatTensor)
        return mask.detach()

    def classify_audio(self, audio):
        prob_a, prob_t = self.prior_a(audio)
        prob_t = self.linear_pool(prob_t.permute(0, 2, 1).contiguous())
        prob_t = prob_t.permute(0, 2, 1).contiguous().view(-1, prob_a.shape[-1])
        return prob_a, prob_t

    def classify_video(self, visual, rois):
        prior = self.prior_v(visual, 'bottom')
        rois = self.align(prior, rois)
        prob, _ = self.prior_v(prior, 'cont')
        prob_r = self.prior_v(rois, 'top')
        prob = torch.softmax(prob, 1)
        prob_r = torch.softmax(prob_r, 1)
        return prob, prob_r

    def save_gradient(self, grad):
        self.gradient.append(grad)

    def forward(self, audio, visual, rois, cluster_a, cluster_v, training=True):
#        audio(batchsize*mix, 1, T, F)
#        visual(batchsize*mix*frame, 3, H, W)
#        rois(batchsize, mix*frame, rois, 4)
        self.gradient = []
        prob_a, prob_t = self.classify_audio(audio)
        prob_v, prob_r = self.classify_video(visual, rois)
        # return prob_a, prob_t, prob_v, prob_r
        label_a, label_v, label_t, label_r = self.generate_label(prob_a, prob_t, prob_v, prob_r, cluster_a, cluster_v)
        mask = self.sample_prior(label_t, label_v)
        feat_v = self.visual(visual, 'bottom')
        feat_a = self.audio(audio, 'embed')
        feat_a.register_hook(self.save_gradient)
        feat_r = self.align(feat_v, rois)
        pred_r = self.visual(feat_r, 'top')
        pred_v, cam_v = self.visual(feat_v, 'cont')
        pred_a, pred_t = self.audio(feat_a, 'cont')
        if not training:
            return pred_a, feat_a, pred_v, feat_v
        discrim = self.discrim(feat_a, feat_v)
        return discrim, mask, pred_a, [pred_v, pred_r], label_a, [label_v, label_r], [feat_a, feat_v], cam_v
        
class MapLoss(nn.Module):
    
    def __init__(self, ):
        super(MapLoss, self).__init__()
        self.bceloss = nn.BCELoss(size_average=False, reduce=False)
        self.celoss = nn.CrossEntropyLoss(size_average=False, reduce=False)
        
    def forward(self, label_a, label_v, label_r, pred_a, pred_v, pred_r):
        aloss = self.bceloss(pred_a, label_a)
        vloss = self.bceloss(torch.sigmoid(pred_v), label_v)
        # rloss = self.bceloss(torch.sigmoid(pred_r), label_r)
        rloss = self.celoss(pred_r, label_r)
        aloss = torch.sum(aloss)/float(aloss.shape[0]) 
        vloss = torch.sum(vloss)/float(vloss.shape[0])
        rloss = torch.sum(rloss)/float(rloss.shape[0])
        return aloss, vloss, rloss
        
class DiscrimHead(nn.Module):
    
    def __init__(self, mix, frames, layer, channel):
        super(DiscrimHead, self).__init__()
        self.mix = mix
        self.frames = frames
        self.temp_conv = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(3, 1), dilation=(2, 1), padding=(2, 0), stride=1, bias=False),
                nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=(1, 2), padding=0, stride=(1, 2), bias=False),
                nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=(3, 1), padding=(1, 0), stride=1, bias=False),
                nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0),
                nn.ReLU(True)
        )
        self.spa_conv = layer
        self.spa_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.temp_pool = nn.AdaptiveMaxPool2d((self.frames, 1))
        self.channel = channel
        self.transform_a = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128)
        )
        self.transform_v = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128)
        )
        self.discrim = nn.Sequential(
            nn.Linear(2 * self.channel, 128),
            nn.ReLU(True),
            nn.Linear(128, 2)
        )

    def save_grad(self, grad):
        self.gradient.append(grad)

    def auto_align(self, embed):
        final = F.mse_loss(self.transform_a(embed[:, :512]),
                            self.transform_v(embed[:, 512:]), reduce=False)
        final = final.mean(1)
        return final

    def forward(self, audio, visual, cam=False):
        self.gradient = []
        feat_a = self.temp_conv(audio)
        feat_v = self.spa_conv(visual)
        feat_v.register_hook(self.save_grad)
        embed_a = self.temp_pool(feat_a)#(batchsize*mix, channel, frame, 1)
        embed_v = self.spa_pool(feat_v)#(batchsize*mix*frame, channel, 1, 1)
        embed_a = embed_a.view(-1, 512, self.frames).permute(0, 2, 1).contiguous()
        embed_v = embed_v.view(-1, self.frames, embed_v.shape[1])
        embed_co = torch.cat([embed_a, embed_v], -1)
        common = self.discrim(embed_co)
#        embed_co = torch.cat([embed_a, embed_v], -1).view(-1, 1024)
#        common = F.mse_loss(self.transform_a(embed_co[:, :512]),
#                            self.transform_v(embed_co[:, 512:]), reduce=False)
#        common = common.mean(1)
        if self.mix == 1:
            differ = None
        else:
            segments = embed_a.shape[0]
            differ = [None] * segments
            if self.frames == 1:
                for seg in range(segments):
                    rank = seg + 1 if seg % 2 == 0 else seg-1
                    differ[seg] = torch.cat([embed_a[seg, :, :], embed_v[rank, :, :]], -1)
            else:
                for seg in range(segments):
                    randidx = torch.randint(0, segments-1, (self.frames,))
                    randidx[randidx>=seg] = randidx[randidx>=seg] + 1
                    randidx = randidx.type(torch.LongTensor)
                    differ[seg] = torch.cat([embed_a[seg, :, :],
                                             embed_v[randidx, torch.randperm(self.frames), :]], -1)
            embed_di = torch.stack(differ, 0)
            differ = self.discrim(embed_di)
#            embed_di = torch.stack(differ, 0).view(-1, 1024)
#            differ = F.mse_loss(self.transform_a(embed_di[:, :512]),
#                                self.transform_v(embed_di[:, 512:]), reduce=False)
#            differ = differ.mean(1)

        return common, differ, feat_a, feat_v

class DiscrimLoss(nn.Module):
    
    def __init__(self, margin):
        super(DiscrimLoss, self).__init__()
        self.margin = margin
        self.celoss = nn.CrossEntropyLoss(size_average=False, reduce=False)

    def contrastive_loss(self, common, differ, mask):
        common = common.view(-1)
        differ = differ.view(-1)
        mask = mask.view(-1)
        if mask is None:
            mask = torch.ones_like(common)
        closs = common
        dloss = torch.relu(self.margin-differ)
        closs = torch.sum(closs * mask) / float(torch.sum(mask))
        dloss = torch.sum(dloss * mask) / float(torch.sum(mask))
        loss = closs + dloss
        return loss

    def cross_entropy_loss(self, common, differ, mask):
        common = common.view(-1, 2)
        differ = differ.view(-1, 2)
        gt_co = torch.ones_like(common[:, 0])
        gt_di = torch.zeros_like(differ[:, 0])
        pred = torch.cat([common, differ], 0)
        gt = torch.cat([gt_co, gt_di], 0)
        loss = self.celoss(pred, gt.type(torch.cuda.LongTensor))
        if mask is None:
            mask = torch.ones_like(pred[:, 0])
        else:
            mask = mask.view(-1).repeat([2])
        loss = loss * mask
        loss = torch.sum(loss) / float(torch.sum(mask))
        return loss

    def forward(self, common, differ, mask):
#        loss = self.contrastive_loss(common, differ, mask)
        loss = self.cross_entropy_loss(common, differ, mask)
        return loss