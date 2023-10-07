import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GConv(nn.Module):
    def __init__(self, in_features, out_features, mode='2d', stride=1):
        super(GConv, self).__init__()
        if mode=='2d':
            self.convs = nn.Sequential(
                nn.Conv2d(in_features, out_features//2, 3, stride, 1),
                nn.BatchNorm2d(out_features//2),
                nn.ReLU(True),
                nn.Conv2d(out_features//2, out_features, 3, 1, 1),
                nn.BatchNorm2d(out_features),
                nn.ReLU(True),
            )
        elif mode == '1d':
            self.convs = nn.Sequential(
                nn.Conv1d(in_features, out_features//2, 3, stride, 1),
                nn.BatchNorm1d(out_features//2),
                nn.ReLU(True),
                nn.Conv1d(out_features//2, out_features, 3, 1, 1),
                nn.BatchNorm1d(out_features),
                nn.ReLU(True),
            )

    def forward(self, input):
        out = self.convs(input)
        return out



class GCN(nn.Module):
    def __init__(self, nfeat, ndims, dropout, out_ch):
        super(GCN, self).__init__()

        self.layers_node_a = nn.Sequential(
            GConv(nfeat*2, ndims[0], mode='1d'),
            nn.MaxPool1d(3, 2, 1),
            GConv(ndims[0], ndims[1], mode='1d'),
            nn.MaxPool1d(3, 2, 1),
            GConv(ndims[1], ndims[2], mode='1d'),
            nn.MaxPool1d(3, 2, 1),
            GConv(ndims[2], ndims[3], mode='1d'),
            nn.MaxPool1d(3, 2, 1),
        )
        self.layers_edge_a = nn.Sequential(
            GConv(nfeat, ndims[0], mode='2d'),
            nn.MaxPool2d(3, 2, 1),
            GConv(ndims[0], ndims[1], mode='2d'),
            nn.MaxPool2d(3, 2, 1),
            GConv(ndims[1], ndims[2], mode='2d'),
            nn.MaxPool2d(3, 2, 1),
            GConv(ndims[2], ndims[3], mode='2d'),
            nn.MaxPool2d(3, 2, 1),
        )


        self.layers_node_g = nn.Sequential(
            GConv(nfeat*2, ndims[0], mode='1d', stride=2),
            nn.MaxPool1d(3, 2, 1),
            GConv(ndims[0], ndims[1], mode='1d', stride=2),
            nn.MaxPool1d(3, 2, 1),
            GConv(ndims[1], ndims[2], mode='1d'),
            nn.MaxPool1d(3, 2, 1),
            GConv(ndims[2], ndims[3], mode='1d'),
            nn.MaxPool1d(3, 2, 1),
        )

        self.layers_edge_g = nn.Sequential(
            GConv(nfeat, ndims[0], mode='2d', stride=2),
            nn.MaxPool2d(3, 2, 1),
            GConv(ndims[0], ndims[1], mode='2d', stride=2),
            nn.MaxPool2d(3, 2, 1),
            GConv(ndims[1], ndims[2], mode='2d'),
            nn.MaxPool2d(3, 2, 1),
            GConv(ndims[2], ndims[3], mode='2d'),
            nn.MaxPool2d(3, 2, 1),
        )

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(ndims[-1]*6, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, out_ch),
            nn.Sigmoid()
        )




        
    def feature_extract_a(self, node, edge):
        node = torch.permute(node, (0, 2, 1))
        edge = torch.permute(edge, (0,3,1,2))

        f_node = self.layers_node_a(node)
        f_edge = self.layers_edge_a(edge)
        
        f_node = F.adaptive_avg_pool1d(f_node, 1).flatten(start_dim=1)
        f_edge = F.adaptive_avg_pool2d(f_edge, 1).flatten(start_dim=1)

        out_f = torch.cat([f_node, f_edge], dim=1)
        return out_f
    
    def feature_extract_g(self, node, edge):
        node = torch.permute(node, (0, 2, 1))
        edge = torch.permute(edge, (0,3,1,2))

        f_node = self.layers_node_g(node)
        f_edge = self.layers_edge_g(edge)
        
        f_node = F.adaptive_avg_pool1d(f_node, 1).flatten(start_dim=1)
        f_edge = F.adaptive_avg_pool2d(f_edge, 1).flatten(start_dim=1)

        out_f = torch.cat([f_node, f_edge], dim=1)
        return out_f


    def forward(self, n_h, e_h, n_l, e_l, n_g, e_g):
        f_h = self.feature_extract_a(n_h, e_h)
        f_l = self.feature_extract_a(n_l, e_l)
        f_g = self.feature_extract_g(n_g, e_g)

        f_all = torch.cat([f_h, f_l, f_g], dim=1)

        out = self.head(f_all)

        return out
        pass







if __name__ == '__main__':

    temp_model = GCN(128, [128,256,512,512], 0.5, 5).cuda()
    temp_n = torch.randn(4, 128, 256).cuda()
    temp_a = torch.randn(4, 128,128, 128).cuda()

    temp_n_g = torch.randn(4, 1024, 256).cuda()
    temp_a_g = torch.randn(4, 1024, 1024, 128).cuda()

    temp_out = temp_model(temp_n, temp_a, temp_n, temp_a, temp_n_g, temp_a_g)
    print('test')