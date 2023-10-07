import torch 
import torch.nn as nn


class basic_conv_module(nn.Module):
    def __init__(self, ch_in, ch_out, ker_sizee=3, padding=1):
        super(basic_conv_module, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, ker_sizee, 1, padding),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(True),
            nn.Conv1d(ch_out, ch_out, 1, 1, 1),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(True),
        )
    def forward(self, input):
        out = self.convs(input)
        return  out

class seq_neu_cls(nn.Module):
    def __init__(self, n_feat_in, n_cls_out):
        super(seq_neu_cls, self).__init__()
        
        ## input(batch * channel * seq_length)
        self.encoder_seq_anti = nn.Sequential(
            basic_conv_module(n_feat_in, 16, 16, 8),
            basic_conv_module(16,64,16,8),
            basic_conv_module(64,256,16,8),

            basic_conv_module(256,512),
            nn.AvgPool1d(3, 2, 1),

            basic_conv_module(512,1024),
            nn.AvgPool1d(3, 2, 1),

            basic_conv_module(1024,1024),
            nn.AvgPool1d(3, 2, 1),


            basic_conv_module(1024,512),
            basic_conv_module(512,256),
            basic_conv_module(256,128),
            basic_conv_module(128,64),
            nn.Dropout(0.95),
            basic_conv_module(64,128),
            basic_conv_module(128,256),
            basic_conv_module(256,512),
            basic_conv_module(512,1024),


            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(start_dim=1),


        )

        self.encoder_info = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(True),
            nn.Linear(16, 2),
            nn.ReLU(True)
        )

        self.cls = nn.Sequential(
            nn.Linear(1026, n_cls_out),
            nn.Sigmoid()
        )


        self.dorp = nn.Dropout(0.5)

    def forward(self, input_seq, input_info):
        f_seq = self.encoder_seq_anti(input_seq[:,2:,:])
        f_inf = self.encoder_info(input_info)
        # f_inf = input_info


        f_all = torch.cat([f_inf, f_seq], dim=1)

        if self.training:
            f_all = self.dorp(f_all)
        cls = self.cls(f_all)
        return cls


class basic_fc_res_module(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(basic_fc_res_module, self).__init__()
        self.ch_align = nn.Sequential(
            nn.Linear(ch_in, ch_out),
        )
        
        self.feat = nn.Sequential(
            nn.Linear(ch_out, ch_out*2),
            nn.BatchNorm1d(ch_out*2),
            nn.ReLU(True),
            nn.Linear(ch_out*2, ch_out),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(True),
        )
    def forward(self, input):
        input = self.ch_align(input)
        out = self.feat(input) + input
        return out

class seq_fc(nn.Module):
    def __init__(self, n_feat_in, n_cls_out):
        super(seq_fc, self).__init__()

        self.encoder_CDRH = nn.Sequential(
            nn.Linear(160, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(True),


            basic_fc_res_module(320,640),

            basic_fc_res_module(640,1024),

            basic_fc_res_module(1024, 512),

            basic_fc_res_module(512, 256),

            basic_fc_res_module(256, 128),


        )
        self.encoder_CDRL = nn.Sequential(
            nn.Linear(160, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(True),


            basic_fc_res_module(320,640),

            basic_fc_res_module(640,1024),

            basic_fc_res_module(1024, 512),

            basic_fc_res_module(512, 256),

            basic_fc_res_module(256, 128),


        )

        self.cls = nn.Sequential(
            nn.Linear(258, n_cls_out),
            nn.Sigmoid()
        )
        self.encoder_info = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(True),
            nn.Linear(16, 2),
        )

        self.dorp = nn.Dropout(0.95)

    def forward(self, input_seq, input_info):
        f_seq_H = self.encoder_CDRH(input_seq[:,0,:])
        f_seq_L = self.encoder_CDRL(input_seq[:,1,:])
        f_inf = self.encoder_info(input_info)
        if self.training:
            f_seq_H = self.dorp(f_seq_H)
            f_seq_L = self.dorp(f_seq_L)


        # f_all = torch.cat([f_seq_H*f_inf[:,:1],  f_seq_L*f_inf[:,1:]], dim=1)
        f_all = torch.cat([f_inf, f_seq_H, f_seq_L], dim=1)

        
        cls = self.cls(f_all)
        return cls


# 'CDRH3','CDRL3'
class seq_neu_cls_tf(nn.Module):
    def __init__(self, n_feat_in, n_cls_out):
        super(seq_neu_cls_tf, self).__init__()
        
        ## input(batch * channel * seq_length)
        self.encoder_seq = nn.Sequential(
            basic_conv_module(n_feat_in, 16),
            basic_conv_module(16,64),
            basic_conv_module(64,256),
        )

        self.encoder_info = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(True),
            nn.Linear(16, 2),
            nn.ReLU(True)
        )

        self.cls = nn.Sequential(
            nn.Linear(258, n_cls_out),
            nn.Sigmoid(),
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        layer_norm = nn.LayerNorm(256)
        self.encoder_tf = nn.TransformerEncoder(encoder_layer, num_layers=8, norm=layer_norm)

        self.dorp = nn.Dropout(0.5)

    def forward(self, input_seq, input_info):
        f_seq = self.encoder_seq(input_seq)
        f_inf = self.encoder_info(input_info)

        f_seq = torch.permute(f_seq, [0,2,1])
        f_seq_tfs = self.encoder_tf(f_seq)
        f_seq_tfs = torch.mean(f_seq_tfs, dim=1)

        # f_seq_tfs = f_seq_tfs[:,0,:]


        f_all = torch.cat([f_seq_tfs, f_inf], dim=1)

        # if self.training:
        #     f_all = self.dorp(f_all)
        cls = self.cls(f_all)
        return cls


class seq_conv(nn.Module):
    def __init__(self, n_cls_out):
        super(seq_conv, self).__init__()
        
        ## input(batch * channel * seq_length)
        self.encoder_seq_anti = nn.Sequential(
            basic_conv_module(4, 16, 16, 8),
            basic_conv_module(16,64,16,8),
            basic_conv_module(64,256,16,8),

            basic_conv_module(256,512),
            nn.AvgPool1d(3, 2, 1),

            basic_conv_module(512,1024),
            nn.AvgPool1d(3, 2, 1),

            basic_conv_module(1024,1024),
            nn.AvgPool1d(3, 2, 1),


            # basic_conv_module(1024,512),
            # # nn.AvgPool1d(3, 2, 1),
            # basic_conv_module(512,256),
            # # nn.AvgPool1d(3, 2, 1),
            # basic_conv_module(256,128),
            # # nn.AvgPool1d(3, 2, 1),
            # basic_conv_module(128,64),
            # basic_conv_module(64,128),
            # basic_conv_module(128,256),
            # basic_conv_module(256,512),
            # basic_conv_module(512,1024),


            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(start_dim=1),


        )

        self.encoder_seq_angen = nn.Sequential(
            basic_conv_module(1, 16, 16, 8),
            basic_conv_module(16,64,16,8),
            basic_conv_module(64,256,16,8),

            basic_conv_module(256,512),
            nn.AvgPool1d(3, 2, 1),

            basic_conv_module(512,1024),
            nn.AvgPool1d(3, 2, 1),

            basic_conv_module(1024,1024),
            nn.AvgPool1d(3, 2, 1),


            basic_conv_module(1024,512),
            nn.AvgPool1d(3, 2, 1),
            basic_conv_module(512,256),
            nn.AvgPool1d(3, 2, 1),
            basic_conv_module(256,128),
            nn.AvgPool1d(3, 2, 1),
            basic_conv_module(128,64),
            basic_conv_module(64,128),
            basic_conv_module(128,256),
            basic_conv_module(256,512),
            basic_conv_module(512,1024),


            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(start_dim=1),


        )

        

        self.cls = nn.Sequential(
            nn.Linear(1024, n_cls_out),
            nn.Sigmoid()
        )


        self.dorp = nn.Dropout(0.85)

    def forward(self, input_seq, input_seq_angen):
        f_seq_anti = self.encoder_seq_anti(input_seq)
        f_seq_angen = self.encoder_seq_angen(input_seq_angen)

        f_seq_all = f_seq_anti + f_seq_angen

        if self.training:
            f_seq_all = self.dorp(f_seq_all)
        cls = self.cls(f_seq_all)
        return cls


