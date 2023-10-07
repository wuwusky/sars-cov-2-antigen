import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def figure_it(predicted_expressions, expressions, figure_name):
    # 画图看一下效果
    model_name = figure_name
    fig_file = model_name + "_prediction_performance"
    # fig = plt.figure(figsize=(4, 4), dpi=300, facecolor='w', edgecolor='k')
    # fig.tight_layout(pad=1)
    x = predicted_expressions
    y = expressions
    r = scipy.stats.pearsonr(x, y)
    sns.regplot(x=x, y=y,
                # scatter_kws={'s': 1, 'linewidth': 0, 'rasterized': True},
                # line_kws={'linewidth': 2},
                color='blue', 
                # robust=1,
                )
    ax = plt.gca()
    # ax.get_legend().remove()
    ax.set_xlabel("predicted")
    ax.set_ylabel("Measured")
    if (r[1] == 0.0):
        ax.set_title(f"PCC = {r[0] : 0.3f} | P < {np.nextafter(0, 1) : 0.0E} | N = {len(x)}")
    else:
        ax.set_title(f"PCC = {r[0] : 0.3f} | P = {r[1] : 0.2E} | N = {len(x)}")
    plt.setp(ax.artists, edgecolor='k')
    plt.setp(ax.lines, color='k')
    # plt.setp(ax.lines, linewidth=1.5)
    plt.yticks(ticks=[0,1,2,3,4,5])
    plt.xticks(ticks=[0,1,2,3,4,5])
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.autoscale(enable=True, axis='y', tight=True)
    ax.set_xlim(xmin=min(x)-0.1, xmax=max(x)+0.1)
    ax.set_ylim(ymin=min(y)-0.1, ymax=max(y)+0.1)
    # plt.savefig("%s.svg" % (fig_file,), bbox_inches="tight")
    # plt.savefig("%s.pdf" % (fig_file,), bbox_inches="tight")
    plt.savefig("%s.png" % (fig_file,), bbox_inches="tight")
    # plt.show()


def eval_fig_s2_a():
    from model_resnet_s2 import resnet18_mt
    from train_config_s2 import dataset_anti_mt
    model_save_dir = './user_data/model_data_s2/'
    model_eval = resnet18_mt()
    model_eval.load_state_dict(torch.load(model_save_dir + '/model_aff_mt_str_best.pth', map_location='cpu'), strict=True)
    model_eval.eval()
    model_eval = model_eval.to(device)

    
    batch_size = 4
    num_sample = 30 ##%
    data_dir = './user_data/data_s2/Affinity_train_data.csv'
    data_ex_dir = './user_data/data_s2/Affinity_train_data_ex.csv'


    data = pd.read_csv(data_dir)
    data_seq = data['Sequence'].values.tolist()
    data_lbl = data['Label'].values.tolist()

    data_ex = pd.read_csv(data_ex_dir)
    data_seq_ex = data_ex['Sequence'].values.tolist()
    data_lbl_ex = data_ex['Label'].values.tolist()

    data_all = data_seq + data_seq_ex
    data_lbl = data_lbl + data_lbl_ex
    data_seq_train = []
    data_lbl_train = []
    data_seq_valid = []
    data_lbl_valid = []

    for i in range(len(data_all)):
        if (i + 1) % (100//num_sample) == 0:
            data_seq_valid.append(data_all[i])
            data_lbl_valid.append(data_lbl[i])
            
        else:
            data_seq_train.append(data_all[i])
            data_lbl_train.append(data_lbl[i])



    dataset_train = dataset_anti_mt(data_seq_train[:], data_lbl_train , 'valid')
    dataset_valid = dataset_anti_mt(data_seq_valid[:], data_lbl_valid , 'valid')
    print('dataset init finished')

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)


    list_labels = []
    list_preds = []
    for data in tqdm(loader_train, ncols=50):
        data_seq,data_seq_angen, data_lbl, data_lbl_oh = data
        data_seq,data_seq_angen, data_lbl, data_lbl_oh = data_seq.to(device),data_seq_angen.to(device), data_lbl.to(device), data_lbl_oh.to(device)

        with torch.no_grad():
            out = model_eval.forward(data_seq, data_seq_angen)


        list_labels += data_lbl.cpu().numpy().tolist()
        temp_preds = torch.sum(out.detach(), dim=1).cpu().view(-1).numpy()
        list_preds += temp_preds.tolist()

    figure_it(list_preds, list_labels, 'Affinity Train')
    
    list_labels = []
    list_preds = []
    for data in tqdm(loader_valid, ncols=50):
        data_seq,data_seq_angen, data_lbl, data_lbl_oh = data
        data_seq,data_seq_angen, data_lbl, data_lbl_oh = data_seq.to(device),data_seq_angen.to(device), data_lbl.to(device), data_lbl_oh.to(device)

        with torch.no_grad():
            out = model_eval.forward(data_seq, data_seq_angen)


        list_labels += data_lbl.cpu().numpy().tolist()
        temp_preds = torch.sum(out.detach(), dim=1).cpu().view(-1).numpy()
        list_preds += temp_preds.tolist()

    figure_it(list_preds, list_labels, 'Affinity Valid')

def eval_fig_s2_n():
    from model_resnet_s2 import resnet18_mt
    from train_config_s2 import dataset_anti_mt
    model_save_dir = './user_data/model_data_s2/'
    model_eval = resnet18_mt()
    model_eval.load_state_dict(torch.load(model_save_dir + '/model_neu_mt_str_best.pth', map_location='cpu'), strict=True)
    model_eval.eval()
    model_eval = model_eval.to(device)

    num_sample = 30 ##%
    batch_size = 4
    data_dir = './user_data/data_s2/Neutralization_train_data.csv'

    data = pd.read_csv(data_dir)
    data_seq = data['Sequence'].values.tolist()
    data_lbl = data['Label'].values.tolist()

    data_all = data_seq
    data_lbl = data_lbl



    data_seq_train = []
    data_lbl_train = []
    data_seq_valid = []
    data_lbl_valid = []

    for i in range(len(data_all)):
        if (i + 1) % (100//num_sample) == 0:
            data_seq_valid.append(data_all[i])
            data_lbl_valid.append(data_lbl[i])
            
        else:
            data_seq_train.append(data_all[i])
            data_lbl_train.append(data_lbl[i])





    dataset_train = dataset_anti_mt(data_seq_train[:], data_lbl_train , 'valid')
    dataset_valid = dataset_anti_mt(data_seq_valid[:], data_lbl_valid , 'valid')
    print('dataset init finished')

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)


    list_labels = []
    list_preds = []
    for data in tqdm(loader_train, ncols=50):
        data_seq,data_seq_angen, data_lbl, data_lbl_oh = data
        data_seq,data_seq_angen, data_lbl, data_lbl_oh = data_seq.to(device),data_seq_angen.to(device), data_lbl.to(device), data_lbl_oh.to(device)

        with torch.no_grad():
            out = model_eval.forward(data_seq, data_seq_angen)


        list_labels += data_lbl.cpu().numpy().tolist()
        temp_preds = torch.sum(out.detach(), dim=1).cpu().view(-1).numpy()
        list_preds += temp_preds.tolist()

    figure_it(list_preds, list_labels, 'Neutralization Train')
    
    list_labels = []
    list_preds = []
    for data in tqdm(loader_valid, ncols=50):
        data_seq,data_seq_angen, data_lbl, data_lbl_oh = data
        data_seq,data_seq_angen, data_lbl, data_lbl_oh = data_seq.to(device),data_seq_angen.to(device), data_lbl.to(device), data_lbl_oh.to(device)

        with torch.no_grad():
            out = model_eval.forward(data_seq, data_seq_angen)


        list_labels += data_lbl.cpu().numpy().tolist()
        temp_preds = torch.sum(out.detach(), dim=1).cpu().view(-1).numpy()
        list_preds += temp_preds.tolist()

    figure_it(list_preds, list_labels, 'Neutralization Valid')



def eval_fig_s3_a():
    from model_G import GCN
    from train_config_s3 import dataset_anti_graph
    model_save_dir = './user_data/model_data_s3/'
    model_eval = GCN(128, [128,256,512,512], 0.5, 5)
    model_eval.load_state_dict(torch.load(model_save_dir + '/model_aff_gcn_best.pth', map_location='cpu'), strict=True)
    model_eval.eval()
    model_eval = model_eval.to(device)

    
    batch_size = 2
    num_sample = 30 ##%
    data_dir = './user_data/data_s2/Affinity_train_data.csv'
    data_ex_dir = './user_data/data_s2/Affinity_train_data_ex.csv'


    data = pd.read_csv(data_dir)
    data_seq = data['Sequence'].values.tolist()
    data_lbl = data['Label'].values.tolist()

    data_ex = pd.read_csv(data_ex_dir)
    data_seq_ex = data_ex['Sequence'].values.tolist()
    data_lbl_ex = data_ex['Label'].values.tolist()

    data_all = data_seq + data_seq_ex
    data_lbl = data_lbl + data_lbl_ex
    data_seq_train = []
    data_lbl_train = []
    data_seq_valid = []
    data_lbl_valid = []

    for i in range(len(data_all)):
        if (i + 1) % (100//num_sample) == 0:
            data_seq_valid.append(data_all[i])
            data_lbl_valid.append(data_lbl[i])
            
        else:
            data_seq_train.append(data_all[i])
            data_lbl_train.append(data_lbl[i])



    dataset_train = dataset_anti_graph(data_seq_train[:], data_lbl_train , 'valid')
    dataset_valid = dataset_anti_graph(data_seq_valid[:], data_lbl_valid , 'valid')
    print('dataset init finished')

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)


    # list_labels = []
    # list_preds = []
    # for data in tqdm(loader_train, ncols=50):
    #     n_h, e_h, n_l, e_l, n_g, e_g, lbl, lbl_oh = data
        
    #     n_h, e_h = n_h.to(device), e_h.to(device)
    #     n_l, e_l = n_l.to(device), e_l.to(device)
    #     n_g, e_g = n_g.to(device), e_g.to(device)
    #     lbl, lbl_oh = lbl.to(device), lbl_oh.to(device)
        
    #     with torch.no_grad():
    #         out = model_eval.forward(n_h, e_h, n_l, e_l, n_g, e_g)


    #     list_labels += lbl.cpu().numpy().tolist()
    #     temp_preds = torch.sum(out.detach(), dim=1).cpu().view(-1).numpy()
    #     list_preds += temp_preds.tolist()

    # figure_it(list_preds, list_labels, 'Affinity Train')
    
    list_labels = []
    list_preds = []
    for data in tqdm(loader_valid, ncols=50):
        n_h, e_h, n_l, e_l, n_g, e_g, lbl, lbl_oh = data
        
        n_h, e_h = n_h.to(device), e_h.to(device)
        n_l, e_l = n_l.to(device), e_l.to(device)
        n_g, e_g = n_g.to(device), e_g.to(device)
        lbl, lbl_oh = lbl.to(device), lbl_oh.to(device)
        
        with torch.no_grad():
            out = model_eval.forward(n_h, e_h, n_l, e_l, n_g, e_g)


        list_labels += lbl.cpu().numpy().tolist()
        temp_preds = torch.sum(out.detach(), dim=1).cpu().view(-1).numpy()
        list_preds += temp_preds.tolist()

    figure_it(list_preds, list_labels, 'Affinity Valid')

def eval_fig_s3_n():
    from model_G import GCN
    from train_config_s3 import dataset_anti_graph
    model_save_dir = './user_data/model_data_s3/'
    model_eval = GCN(128, [128,256,512,512], 0.5, 5)
    model_eval.load_state_dict(torch.load(model_save_dir + '/model_neu_gcn_best.pth', map_location='cpu'), strict=True)
    model_eval.eval()
    model_eval = model_eval.to(device)

    num_sample = 30 ##%
    batch_size = 2
    data_dir = './user_data/data_s2/Neutralization_train_data.csv'

    data = pd.read_csv(data_dir)
    data_seq = data['Sequence'].values.tolist()
    data_lbl = data['Label'].values.tolist()

    data_all = data_seq
    data_lbl = data_lbl



    data_seq_train = []
    data_lbl_train = []
    data_seq_valid = []
    data_lbl_valid = []

    for i in range(len(data_all)):
        if (i + 1) % (100//num_sample) == 0:
            data_seq_valid.append(data_all[i])
            data_lbl_valid.append(data_lbl[i])
            
        else:
            data_seq_train.append(data_all[i])
            data_lbl_train.append(data_lbl[i])





    dataset_train = dataset_anti_graph(data_seq_train[:], data_lbl_train , 'valid')
    dataset_valid = dataset_anti_graph(data_seq_valid[:], data_lbl_valid , 'valid')
    print('dataset init finished')

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)


    # list_labels = []
    # list_preds = []
    # for data in tqdm(loader_train, ncols=50):
    #     n_h, e_h, n_l, e_l, n_g, e_g, lbl, lbl_oh = data
        
    #     n_h, e_h = n_h.to(device), e_h.to(device)
    #     n_l, e_l = n_l.to(device), e_l.to(device)
    #     n_g, e_g = n_g.to(device), e_g.to(device)
    #     lbl, lbl_oh = lbl.to(device), lbl_oh.to(device)
        
    #     with torch.no_grad():
    #         out = model_eval.forward(n_h, e_h, n_l, e_l, n_g, e_g)


    #     list_labels += lbl.cpu().numpy().tolist()
    #     temp_preds = torch.sum(out.detach(), dim=1).cpu().view(-1).numpy()
    #     list_preds += temp_preds.tolist()

    # figure_it(list_preds, list_labels, 'Neutralization Train')
    
    list_labels = []
    list_preds = []
    for data in tqdm(loader_valid, ncols=50):
        n_h, e_h, n_l, e_l, n_g, e_g, lbl, lbl_oh = data
        
        n_h, e_h = n_h.to(device), e_h.to(device)
        n_l, e_l = n_l.to(device), e_l.to(device)
        n_g, e_g = n_g.to(device), e_g.to(device)
        lbl, lbl_oh = lbl.to(device), lbl_oh.to(device)
        
        with torch.no_grad():
            out = model_eval.forward(n_h, e_h, n_l, e_l, n_g, e_g)


        list_labels += lbl.cpu().numpy().tolist()
        temp_preds = torch.sum(out.detach(), dim=1).cpu().view(-1).numpy()
        list_preds += temp_preds.tolist()

    figure_it(list_preds, list_labels, 'Neutralization Valid')



if __name__ == '__main__':
    # eval_fig_s2_a()
    # eval_fig_s2_n()

    eval_fig_s3_a()
    # eval_fig_s3_n()