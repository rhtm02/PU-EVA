import torch
import torch.nn as nn
import pointops.functions as pointops
from models.util import PixelShuffle1D


class FeatureExtractionComponent(nn.Module):
    def __init__(self,pre_channel,num_neighbors,G=12):
        super(FeatureExtractionComponent, self).__init__()

        self.num_neighbors = num_neighbors
        self.pre = nn.Sequential(nn.Conv1d(pre_channel,24,1),
                                 nn.BatchNorm1d(24),
                                 nn.ReLU(inplace=True))

        self.grouper = pointops.QueryAndGroup(nsample=num_neighbors + 1,return_idx=True,use_xyz=False)

        in_channel = 48
        self.shared_mlp_1 = nn.Sequential(nn.Conv2d(in_channel,G,1,bias=False),
                                          nn.BatchNorm2d(G),
                                          nn.ReLU(inplace=True))
        in_channel += G
        self.shared_mlp_2 = nn.Sequential(nn.Conv2d(in_channel,G,1,bias=False),
                                      nn.BatchNorm2d(G),
                                      nn.ReLU(inplace=True))
        in_channel += G
        self.shared_mlp_3 = nn.Conv2d(in_channel, G,1,bias=False)

        self.maxpool = nn.MaxPool2d([1,self.num_neighbors])

    def forward(self,x,xyz=None):
        '''
        :param x: [B,C,N]
        :return: [B,C',N]
        '''
        if xyz == None:
            xyz = x.detach()
        # pre layer [B,24,N]
        pre_points = self.pre(x)
        # knn [B,24,N,K]
        grouped_feature,grouped_xyz,indices = self.grouper(xyz=xyz.permute(0,2,1).contiguous(),features=pre_points)
        grouped_feature = grouped_feature[...,1:]
        grouped_xyz = grouped_xyz[...,1:]
        indices = indices[...,1:]
        #print(grouped_feature.shape,grouped_xyz.shape,indices.shape)
        #torch.Size([2, 24, 256, 16]) torch.Size([2, 3, 256, 16]) torch.Size([2, 256, 16])

        y = torch.cat([pre_points.unsqueeze(-1).repeat(1,1,1,self.num_neighbors),grouped_feature],dim=1)
        # print(y.shape) #torch.Size([2, 48, 256, 15])

        y = torch.cat([self.shared_mlp_1(y),y],dim=1) #torch.Size([2, 60, 256, 15])
        # print(y.shape)

        y = torch.cat([self.shared_mlp_2(y),y],dim=1) #torch.Size([2, 72, 256, 15])
        # print(y.shape)

        y = torch.cat([self.shared_mlp_3(y),y],dim=1) #torch.Size([2, 84, 256, 15])
        # print(y.shape)

        y = self.maxpool(y) #torch.Size([2, 84, 256, 1])
        # print(y.shape)

        y = torch.cat([y.squeeze(),x],dim=1)
        return y

class DenseFeatureExtractor(nn.Module):
    def __init__(self,num_neighbors,G=12):
        super(DenseFeatureExtractor, self).__init__()

        self.feature_extractor_1 = FeatureExtractionComponent(pre_channel=3,num_neighbors=num_neighbors,G=G)

        self.feature_extractor_2 = FeatureExtractionComponent(pre_channel=84 + 3, num_neighbors=num_neighbors, G=G)

        self.feature_extractor_3 = FeatureExtractionComponent(pre_channel=84 + 84 + 3, num_neighbors=num_neighbors, G=G)

        self.feature_extractor_4 = FeatureExtractionComponent(pre_channel=84 + 84 + 84 + 3, num_neighbors=num_neighbors, G=G)

    def forward(self,x):
        '''
        :param x: [B,3,N]
        :return: [B,339,N]
        '''
        y = self.feature_extractor_1(x) # torch.Size([2, 87, 256])
        # print(f'FE 1 : {y.shape}')
        y = self.feature_extractor_2(y,x) # torch.Size([2, 171, 256])
        # print(f'FE 2 : {y.shape}')
        y = self.feature_extractor_3(y,x) # torch.Size([2, 255, 256])
        # print(f'FE 3 : {y.shape}')
        y = self.feature_extractor_4(y,x) # torch.Size([2, 339, 256])
        # print(f'FE 4 : {y.shape}')
        return y
class EVAUpsampling(nn.Module):

    def __init__(self,num_neighbors,upscale_factor,in_channel):
        super(EVAUpsampling, self).__init__()
        assert num_neighbors > upscale_factor, 'num_neighbors > upscale_factor'
        self.num_neighbors = num_neighbors
        self.geo_grouper = pointops.QueryAndGroup(nsample=num_neighbors + 1, return_idx=True)
        self.feature_grouper = pointops.QueryAndGroupFeature(nsample=num_neighbors + 1, use_feature=True)
        self.R = upscale_factor

        self.g_linear = nn.Sequential(nn.Conv2d(in_channels=3 * in_channel + 3 * 3 + 1,
                                                out_channels=512,
                                                kernel_size=1,
                                                bias=True),
                                      nn.ReLU(inplace=True))

        self.h_linear = nn.Sequential(nn.Conv2d(in_channels=3 * in_channel + 3 * 3 + 1,
                                                out_channels=512,
                                                kernel_size=1,
                                                bias=True),
                                      nn.ReLU(inplace=True))
        self.l_linear = nn.Sequential(nn.Conv2d(in_channels=3 * in_channel + 3 * 3 + 1,
                                                out_channels=512,
                                                kernel_size=1,
                                                bias=True),
                                      nn.ReLU(inplace=True))

        self.softmax = nn.Softmax(dim=-1)


    def forward(self,xyz,f):
        B,F_DIM,N = f.shape
        xyz_diff, grouped_xyz, indices = self.geo_grouper(xyz=xyz)  # (B, diff, N, K) (B,3,N,K)
        xyz_diff = xyz_diff[...,1:];
        grouped_xyz  = grouped_xyz[...,1:]
        G_feat = self.feature_grouper(xyz=xyz, features=f, idx=indices.int())[...,1:]  # (B,diff + f, N, K)


        xyz_repeat = xyz.permute(0, 2, 1).unsqueeze(-1).repeat(1, 1, 1, self.num_neighbors)
        xyz_dist = torch.sqrt(xyz_diff[:,0,:,:]**2 + xyz_diff[:,1,:,:]**2 + xyz_diff[:,2,:,:]**2).unsqueeze(dim=1)
        G_geo = torch.cat([xyz_diff,grouped_xyz], dim=1)

        # v_i,k - v_i | v_i,k | v_i | p_i,k - p_i | p_i.k | p_i | d_i,k
        F = torch.cat([G_feat,G_feat[:,F_DIM:,:,:] - G_feat[:,:F_DIM,:,:],G_geo,xyz_repeat,xyz_dist],dim=1) # (B, 3 * F_DIM + 3 * 3 + 1, N, K)

        # for random sampling
        r_indice = torch.argsort(torch.rand(B,3 * F_DIM + 3 * 3 + 1,N,self.num_neighbors,device='cuda'))[...,:self.R].cuda()
        r = F.gather(-1,r_indice) # (B, 3 * F_DIM + 3 * 3 + 1, N, R)

        g = self.g_linear(r) # (B, 512, N, R)
        h = self.h_linear(F) # (B, 512, N, K)

        # Similarity Matrix S
        similarity_matrix = self.softmax(torch.matmul(g.permute(0,2,3,1),h.permute(0,2,1,3))) # (B, N, R, K)

        affine_combination = torch.matmul(similarity_matrix,grouped_xyz.permute(0,2,3,1)) # (B, N, R, 3)
        affine_combination = affine_combination.permute(0,3,1,2)  # (B, 3, N, R)

        l = self.l_linear(F) # (B, 512, N, K)
        l_prime = torch.max(l,dim=3)[0].unsqueeze(dim=-1).repeat([1,1,1,self.R]) # (B, 512, N, R)

        output_feature = torch.cat([l_prime,affine_combination],dim=1) # (B, 512 + 3, N, R)
        coarse_xyz = affine_combination.view(B,3,N * self.R) # (B, 3, N * R)

        return coarse_xyz.permute(0,2,1),output_feature







class CoordinateReconstruction(nn.Module):

    def __init__(self,in_channel):
        super(CoordinateReconstruction, self).__init__()

        self.offset_linear = nn.Sequential(nn.Conv2d(in_channel,in_channel,1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(in_channel,in_channel,1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(in_channel, 128, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(128, 3, 1))

    def forward(self,f):
        B,_,N,R = f.shape
        return self.offset_linear(f).view(B,3,N * R).permute(0,2,1)


class PuEVA(nn.Module):

    def __init__(self,num_neighbors):
        super(PuEVA, self).__init__()

        self.feature_extractor = DenseFeatureExtractor(num_neighbors=num_neighbors)

        self.upsampler = EVAUpsampling(num_neighbors=num_neighbors,upscale_factor=4,in_channel=339)

        self.regressor = CoordinateReconstruction(in_channel=515)


    def forward(self,x):

        feature = self.feature_extractor(x)

        coarse, feature = self.upsampler(x.permute(0,2,1).contiguous(), feature)

        offset = self.regressor(feature)
        dens_points = coarse + offset

        sampling_idx = pointops.furthestsampling(dens_points,1024)

        sampling_points = pointops.gathering(dens_points.permute(0,2,1).contiguous(),sampling_idx)

        return dens_points#sampling_points.permute(0,2,1)



if __name__ == '__main__':
    x = torch.randn(63,3,256).cuda()
    f = PuEVA(num_neighbors=12).cuda()
    print(f(x).shape)
