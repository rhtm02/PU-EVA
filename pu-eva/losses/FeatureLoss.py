import torch
import torch.nn as nn
import torch.nn.functional as F

import collections

from models.pointnet2_utils import PointNetSetAbstraction


class PointNet2Feature(nn.Module):
    def __init__(self):
        super(PointNet2Feature, self).__init__()


        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

    def load_params(self,pointnet='./logs/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'):
        checkpoint = torch.load(pointnet)

        state_dict = collections.OrderedDict()
        for name in self.state_dict().keys():
            state_dict[name] = checkpoint['model_state_dict'][name]
        self.load_state_dict(state_dict)

    def forward(self, xyz):

        B, _, _ = xyz.shape

        norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        feature = l3_points.view(B, 1024)

        return feature


class PerceptualLoss(nn.Module):
    def __init__(self,path='./logs/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'):
        super(PerceptualLoss, self).__init__()

        self.features = PointNet2Feature()
        self.features.load_params(path)

        for name, param in self.features.named_parameters():
            param.requires_grad = False

    def forward(self, source, target):

        loss = torch.nn.functional.l1_loss(self.features(source), self.features(target))

        return loss