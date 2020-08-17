import torch
import torch.nn as nn
import torch.nn.functional as F

def filter_prob(cls_a, cls_v, thres):
    assert cls_a.shape == cls_v.shape
    prob_a = F.sigmoid(cls_a).view(-1)
    prob_v = F.sigmoid(cls_v).view(-1)
    eff_a = (prob_a.unsqueeze(1)>thres)
    eff_v = (prob_v.unsqueeze(0)>thres)
    eff = eff_a * eff_v
    eff = eff.type(torch.FloatTensor).to(cls_a.device)
    # eff = eff * (prob_a.unsqueeze(1) * prob_v.unsqueeze(0))
    return eff
    
def contrastive(fine_a, fine_v):
    assert fine_a.shape == fine_v.shape
    assert fine_a.shape[1] == 128
    fine_a = fine_a.permute(0, 2, 1).contiguous().view(-1, 128)
    fine_v = fine_v.permute(0, 2, 1).contiguous().view(-1, 128)
    similarity = torch.mm(fine_a, fine_v.permute(1, 0).contiguous())
    return similarity

class Align(nn.Module):
    
    def __init__(self, vision, audio):
        super(Align, self).__init__()
        self.vision = vision
        self.audio = audio
        self.avc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 2)
        )
        self.project_a = nn.Sequential(
            nn.Conv1d(512, 1024, 1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(1024, 128, 1, bias=False)
        )
        self.project_v = nn.Sequential(
            nn.Conv1d(512, 1024, 1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(1024, 128, 1, bias=False)
        )
        self.class_a = nn.Conv2d(512, 7, 1, bias=False)
        self.class_v = nn.Conv2d(512, 7, 1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, spec, img):
        N = spec.shape[0]
        feat_a = self.audio(spec)
        feat_v = self.vision(img)
        cam_a = F.relu(self.class_a(feat_a)).detach()
        cam_v = F.relu(self.class_v(feat_v)).detach()
        
        fine_a = feat_a.unsqueeze(2) * cam_a.unsqueeze(1)
        fine_v = feat_v.unsqueeze(2) * cam_v.unsqueeze(1)
        weight_a = torch.sum(cam_a.view(*cam_a.shape[:2], -1), -1)
        weight_v = torch.sum(cam_v.view(*cam_v.shape[:2], -1), -1)
        fine_a = torch.mean(fine_a.view(*fine_a.shape[:3], -1), -1)
        fine_v = torch.mean(fine_v.view(*fine_v.shape[:3], -1), -1)
        fine_a = fine_a / (weight_a.unsqueeze(1)+1e-10)
        fine_v = fine_v / (weight_v.unsqueeze(1)+1e-10)
        fine_a = self.project_a(fine_a)
        fine_v = self.project_v(fine_v)
        fine_a = F.normalize(fine_a, p=2, dim=1)
        fine_v = F.normalize(fine_v, p=2, dim=1)
        
        feat_a = self.avgpool(feat_a)
        feat_v = self.avgpool(feat_v)
        
        fusion = torch.cat([feat_a.unsqueeze(1).repeat([1, N, 1, 1, 1]), 
                            feat_v.unsqueeze(0).repeat([N, 1, 1, 1, 1])], 2)
        fusion = torch.flatten(fusion, 2)
        avc = self.avc(fusion)
        
        cls_a = self.class_a(feat_a)
        cls_v = self.class_v(feat_v)
        return avc, cls_a.flatten(1), cls_v.flatten(1), fine_a, fine_v
    
class Location(nn.Module):
    
    def __init__(self, vision, audio):
        super(Location, self).__init__()
        self.vision = vision
        self.audio = audio
        self.avc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 2)
        )
        self.project_a = nn.Sequential(
            nn.Conv1d(512, 1024, 1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(1024, 128, 1, bias=False)
        )
        self.project_v = nn.Sequential(
            nn.Conv1d(512, 1024, 1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(1024, 128, 1, bias=False)
        )
        self.class_a = nn.Conv2d(512, 7, 1, bias=False)
        self.class_v = nn.Conv2d(512, 7, 1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, spec, img):
        N = spec.shape[0]
        feat_a = self.audio(spec)
        feat_v = self.vision(img)
        cam_a = F.relu(self.class_a(feat_a))
        cam_v = F.relu(self.class_v(feat_v))
        
        fine_a = feat_a.unsqueeze(2) * cam_a.unsqueeze(1)
        weight_a = torch.sum(cam_a.view(*cam_a.shape[:2], -1), -1)
        fine_a = torch.mean(fine_a.view(*fine_a.shape[:3], -1), -1)
        fine_a = fine_a / (weight_a.unsqueeze(1)+1e-10)
        fine_a = self.project_a(fine_a)
        fine_v = self.project_v(feat_v.view(*feat_v.shape[:2], -1))
        fine_a = F.normalize(fine_a, p=2, dim=1)
        fine_v = F.normalize(fine_v, p=2, dim=1)
        
        feat_a = self.avgpool(feat_a)
        feat_v = self.avgpool(feat_v)
        align_a = self.project_a(feat_a.flatten(2))
        align_v = self.project_v(feat_v.flatten(2))
        
        fusion = torch.cat([feat_a.unsqueeze(1).repeat([1, N, 1, 1, 1]), 
                            feat_v.unsqueeze(0).repeat([N, 1, 1, 1, 1])], 2)
        fusion = torch.flatten(fusion, 2)
        avc = self.avc(fusion)
        
        cls_a = self.class_a(feat_a)
        cls_v = self.class_v(feat_v)
        return avc, cls_a.flatten(1), cls_v.flatten(1), fine_a, fine_v, cam_a, cam_v,\
            align_a, align_v