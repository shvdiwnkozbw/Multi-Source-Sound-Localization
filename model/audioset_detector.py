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
        cluster = torch.softmax(-1000 * dist, 1)
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
    inter = torch.max(min_x_max - max_x_min, torch.zeros_like(min_x_max)) * torch.max(min_y_max - max_y_min,
                                                                                      torch.zeros_like(min_y_max))
    union = (max_x_max - min_x_min) * (max_y_max - min_y_min)
    weight = inter / union
    return weight


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
            cam = torch.sum(weights.view(weights.shape[0], weights.shape[1], 1, 1) * target.unsqueeze(1), 1)
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
        self.align = RoIAlignAvg(size, size, 1.0 / self.scale)
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
        common, differ, feat_a, feat_v = self.model.discrim(feat_a, feat_v)
        for idx in range(2):
            one_hot = torch.zeros_like(common.view(-1, 2))
            one_hot[:, idx] = 1
            one_hot = torch.sum(one_hot * common.view(-1, 2))
            self.model.zero_grad()
            one_hot.backward(retain_graph=True)
            grad = self.model.discrim.gradient[-1]
            weights = torch.sum(grad.view(grad.shape[0], grad.shape[1], -1), -1)
            cam = torch.sum(weights.view(weights.shape[0], weights.shape[1], 1, 1) * feat_v, 1)
            cams.append(cam)
        cam = torch.relu(cam)
        norm = torch.max(cam.view(cam.shape[0], -1), 1)[0]
        cam = cam / (norm.view(norm.shape[0], 1, 1)+1e-10)
        # cam = torch.clamp(cam, 0.0, 1.0)
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
        pcam = torch.relu(cam - 0.1)
        pcam[pcam > 0] = 1
        return pcam.view(pcam.shape[0], 1, -1).detach(), \
               torch.sum(pcam.view(pcam.shape[0], -1), -1) / (pcam.shape[1] * pcam.shape[2]), \
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

    def __init__(self, frame, model):
        super(AVAlign, self).__init__()
        self.frame = frame
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

    def forward(self, feat_a, feat_v, label, cam):
        # label_a(mix, class)
        # label_v(mix, class)
        feat_a = self.model.temp_conv(feat_a)
        feat_a = self.temp_pool(feat_a)
        feat_a = feat_a.view(-1, label.shape[-1], feat_a.shape[1])

        feat_v = self.model.spa_conv(feat_v)
        feat_v = self.cam_pool(cam, feat_v)
        feat_v = feat_v.view(-1, self.frame, label.shape[-1], feat_v.shape[-1])

        assert feat_a.shape[0] == feat_v.shape[0] == label.shape[0]
        segments = feat_a.shape[0]
        same = [None] * segments
        differ = [None] * segments
        embed_co = []
        embed_di = []
        index = (label > 0)

        for seg in range(segments):
            same[seg] = []
            differ[seg] = []
            rank = seg + 1 if seg % 2 == 0 else seg - 1
            if torch.sum(index[seg] > 0) == 0:
                continue
            eff_a = torch.arange(index.shape[-1])[index[seg] > 0]
            differ_v = feat_v[rank].permute(1, 0, 2).contiguous()[index[rank] > 0].\
                permute(1, 0, 2).contiguous()
            differ_ind = torch.arange(label.shape[-1])[index[rank] > 0].unsqueeze(0).repeat([self.frame, 1])
            for ind in eff_a:
                same[seg].append(torch.cat([feat_a[seg, ind].unsqueeze(0).repeat([self.frame, 1]),
                                            feat_v[seg, :, ind]], -1))
                rep = torch.sum(differ_ind != ind)
                if rep == 0:
                    continue
                differ[seg].append(torch.cat([feat_a[seg, ind].unsqueeze(0).repeat([rep, 1]),
                                              differ_v[differ_ind != ind]], -1))

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

    def __init__(self, margin=2):
        super(AlignLoss, self).__init__()
        self.margin = 2
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
#        loss = self.cross_entropy_loss(common, differ)
        return loss


class Framework(nn.Module):

    def __init__(self, pretrained, mix, frames):
        super(Framework, self).__init__()
        self.mix = mix
        self.frames = frames
        self.visual = resnet18(pretrained, num_classes=15, additional=True)
        self.audio = mobilecrnn_v2(outputdim=15, pretrained=pretrained)
        self.discrim = DiscrimHead(self.mix, self.frames, self.visual.layer5, 512)
        self.temp_pool = nn.AdaptiveMaxPool1d((self.frames))
        self.avalign = AVAlign(frames, self.discrim)

    def save_gradient(self, grad):
        self.gradient.append(grad)

    def forward(self, audio, visual, training=True):
        self.gradient = []
        feat_v = self.visual(visual, 'bottom')
        feat_a = self.audio(audio, 'embed')
        feat_a.register_hook(self.save_gradient)
        pred_v, cam_v = self.visual(feat_v, 'cont')
        pred_a, _ = self.audio(feat_a, 'cont')
#        return pred_a, pred_v
        if not training:
            return pred_a, pred_v, feat_a, feat_v, cam_v
        discrim = self.discrim(feat_a, feat_v)
        return discrim, pred_a, pred_v, feat_a, feat_v, cam_v

class MapLoss(nn.Module):

    def __init__(self, ):
        super(MapLoss, self).__init__()
        self.bceloss = nn.BCELoss(size_average=False, reduce=False)

    def forward(self, label, pred_a, pred_v):
        aloss = self.bceloss(pred_a, label)
        vloss = self.bceloss(torch.sigmoid(pred_v),
                             label.unsqueeze(1).repeat([1, 4, 1]).view(-1, label.shape[-1]))
        aloss = torch.sum(aloss) / float(aloss.shape[0])
        vloss = torch.sum(vloss) / float(vloss.shape[0])
        return aloss, vloss


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
        self.temp_pool = nn.AdaptiveMaxPool2d((1, 1))
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

    def forward(self, audio, visual):
        self.gradient = []
        feat_a = self.temp_conv(audio)
        feat_v = self.spa_conv(visual)
        feat_v.register_hook(self.save_grad)
        embed_a = self.temp_pool(feat_a)
        embed_a = embed_a.repeat([1, 1, self.frames, 1])
        embed_v = self.spa_pool(feat_v)  # (batchsize*mix*frame, channel, 1, 1)
        embed_a = embed_a.view(-1, 512, self.frames).permute(0, 2, 1).contiguous()
        embed_v = embed_v.view(-1, self.frames, embed_v.shape[1])
        embed_co = torch.cat([embed_a, embed_v], -1)
        common = self.discrim(embed_co)
        if self.mix == 0:
            differ = None
        else:
            segments = embed_a.shape[0]
            differ = [None] * segments
            if self.frames == 1:
                for seg in range(segments):
                    rank = seg + 1 if seg % 2 == 0 else seg - 1
                    differ[seg] = torch.cat([embed_a[seg, :, :], embed_v[rank, :, :]], -1)
            else:
                for seg in range(segments):
                    randidx = torch.randint(0, segments - 1, (self.frames,))
                    randidx[randidx >= seg] = randidx[randidx >= seg] + 1
                    randidx = randidx.type(torch.LongTensor)
                    differ[seg] = torch.cat([embed_a[seg, :, :],
                                             embed_v[randidx, torch.randperm(self.frames), :]], -1)
            embed_di = torch.stack(differ, 0)
            differ = self.discrim(embed_di)
        return common, differ, feat_a, feat_v


class DiscrimLoss(nn.Module):

    def __init__(self):
        super(DiscrimLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss(size_average=False, reduce=False)

    def forward(self, common, differ, mask=None):
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
