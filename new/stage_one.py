import torch
import torch.nn as nn

class MTask(nn.Module):
    
    def __init__(self, vision, audio):
        super(MTask, self).__init__()
        self.vision = vision
        self.audio = audio
        self.avc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 2)
        )
        self.class_a = nn.Conv2d(512, 7, 1, bias=False)
        self.class_v = nn.Conv2d(512, 7, 1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, spec, img):
        N = spec.shape[0]
        feat_a = self.audio(spec)
        feat_v = self.vision(img)
        feat_a = self.avgpool(feat_a)
        feat_v = self.avgpool(feat_v)
        
        fusion = torch.cat([feat_a.unsqueeze(1).repeat([1, N, 1, 1, 1]), 
                            feat_v.unsqueeze(0).repeat([N, 1, 1, 1, 1])], 2)
        fusion = torch.flatten(fusion, 2)
        avc = self.avc(fusion)
        
        cls_a = self.class_a(feat_a)
        cls_v = self.class_v(feat_v)
        return avc, cls_a.flatten(1), cls_v.flatten(1)