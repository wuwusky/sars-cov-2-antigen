import pandas as pd
import  torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# from data_gen_s2 import *
# from model_resnet_s2 import *
from model_G import *
max_len = 128
max_len_ag = 1024
from test_feature_extract import *

logging.info(f'constructing omegafold')
model_omega = of.OmegaFold(of.make_config())
if "model" in state_dict:
    state_dict = state_dict.pop("model")
model_omega.load_state_dict(state_dict)
model_omega.eval()
model_omega.to(args.device)

# from tqdm import tqdm
# from tqdm.contrib import tzip
from collections import Counter

def save_result(list_results_a, list_results_n):
    num_a = len(list_results_a)
    num_b = len(list_results_n)
    num_max = max(num_a, num_b)
    list_lines = []
    list_lines.append('label_a,label_n')
    for i in range(num_max):
        try:
            temp_a = list_results_a[i]
        except Exception as e:
            temp_a = ''
        try:
            temp_b = list_results_n[i]
        except Exception as e:
            temp_b = ''

        list_lines.append(str(temp_a) +','+ str(temp_b))
        
    os.makedirs('./prediction_result', exist_ok=True)
    
    with open('./prediction_result/result.csv', mode='w', encoding='gb2312', newline='') as f:
        for row in list_lines:
            f.write(row+'\n')
    print('test')


def load_test(data_dir):
    with open(data_dir, mode='r', encoding='gb2312') as f:
        pd_info = pd.read_csv(f)
    temp_names_np = pd_info.values[:,0]
    return temp_names_np.tolist()



def predict_seqs_neu(pd_info, dict_antigen_seqs, model_neu):
    seq_feature_names = ['VH or VHH','VL','CDRH3','CDRL3', 'Neutralising Vs']
    seq_infos_np = pd_info[seq_feature_names].values

    
    list_results_n = []
    for seq_info in tqdm(seq_infos_np, ncols=100):

        for n, temp_data_seq in enumerate(seq_info):
            temp_data_seq = str(temp_data_seq)
            if 'I253' in temp_data_seq:
                if '+I253' in temp_data_seq:
                    seq_info[n] = temp_data_seq.split('+I253')[0]
                else:
                    seq_info[n] = temp_data_seq.split('I253')[0]
            elif ' ' in temp_data_seq:
                seq_info[n] = temp_data_seq.split(' ')[0]
            elif 'nan' in temp_data_seq:
                seq_info[n] = temp_data_seq[3:]


        seq_vh, seq_vl, seq_cdrh, seq_cdrl, info_vs = seq_info

        seq_vh = str(seq_vh)
        seq_vl = str(seq_vl)
        seq_cdrh = str(seq_cdrh)
        seq_cdrl = str(seq_cdrl)
        try:
            info_vs = info_vs.split(';')[0]
            seq_angen = str(dict_antigen_seqs[str(info_vs)])
        except Exception as e:
            seq_angen = str(dict_antigen_seqs[str('SARS-CoV2_WT')])

        with torch.no_grad():
            vh = data_process_sim(seq_vh, max_len)
            features = model_omega.feature_extract(vh, forward_config)
            n_h, e_h = features

            vl = data_process_sim(seq_vl, max_len)
            features = model_omega.feature_extract(vl, forward_config)
            n_l, e_l = features

            ag = data_process_sim(seq_angen, max_len_ag)
            features = model_omega.feature_extract(ag, forward_config)
            n_g, e_g = features

            out = model_neu.forward(n_h.unsqueeze(0), e_h.unsqueeze(0), n_l.unsqueeze(0), e_l.unsqueeze(0), n_g.unsqueeze(0), e_g.unsqueeze(0))

        temp_preds = torch.sum(out.detach().cpu(), dim=1).view(-1).numpy()
        list_results_n += (temp_preds+0.5).astype(np.int8).tolist()
        
        
    return list_results_n



def predict_seqs_aff(pd_info, dict_antigen_seqs, model_aff):
    seq_feature_names = ['VH or VHH','VL','CDRH3','CDRL3', 'Binds to']
    seq_infos_np = pd_info[seq_feature_names].values

    list_results_a = []
    for seq_info in tqdm(seq_infos_np, ncols=100):

        for n, temp_data_seq in enumerate(seq_info):
            temp_data_seq = str(temp_data_seq)
            if 'I253' in temp_data_seq:
                if '+I253' in temp_data_seq:
                    seq_info[n] = temp_data_seq.split('+I253')[0]
                else:
                    seq_info[n] = temp_data_seq.split('I253')[0]
            elif ' ' in temp_data_seq:
                seq_info[n] = temp_data_seq.split(' ')[0]
            elif 'nan' in temp_data_seq:
                seq_info[n] = temp_data_seq[3:]


        seq_vh, seq_vl, seq_cdrh, seq_cdrl, info_vs = seq_info

        seq_vh = str(seq_vh)
        seq_vl = str(seq_vl)
        seq_cdrh = str(seq_cdrh)
        seq_cdrl = str(seq_cdrl)
        try:
            info_vs = info_vs.split(';')[0]
            seq_angen = str(dict_antigen_seqs[str(info_vs)])
        except Exception as e:
            seq_angen = str(dict_antigen_seqs[str('SARS-CoV2_WT')])
        

        with torch.no_grad():
            vh = data_process_sim(seq_vh, max_len)
            features = model_omega.feature_extract(vh, forward_config)
            n_h, e_h = features

            vl = data_process_sim(seq_vl, max_len)
            features = model_omega.feature_extract(vl, forward_config)
            n_l, e_l = features

            ag = data_process_sim(seq_angen, max_len_ag)
            features = model_omega.feature_extract(ag, forward_config)
            n_g, e_g = features

            out = model_aff.forward(n_h.unsqueeze(0), e_h.unsqueeze(0), n_l.unsqueeze(0), e_l.unsqueeze(0), n_g.unsqueeze(0), e_g.unsqueeze(0))

        temp_preds = torch.sum(out.detach().cpu(), dim=1).view(-1).numpy()
        list_results_a += (temp_preds+0.5).astype(np.int8).tolist()
        
    return list_results_a



antigen_list = ['SARS-CoV1', 'SARS','SARS-1', 'SARS-CoV1_WT', 'SARS-CoV', 
                'SARS-2', 'SARS-CoV2_WT', 'Original','SARS-CoV2',
                'Alpha','Beta', 'Gamma', 'Delta', 'B.1.1.7', 'B.1.617.2', 'B.1.351', 'P.1', 'beta', 
                'BA.2', 'Kappa', 'delta', 'gamma', 'alpha', 'epsilon', 'Omicron', 'B.1.617.1',
                'MERS-CoV_WT', 'MERS-CoV', 'Mu', 'BA.1','OC43','Eplison',
                # 'Lota',
                'lamnda',
                'WIV-1', 
                # 'Q493R', 'N440K', 'V367F', 'R346S', 'A457V', 'D614G','S477N', 'E484K', 'V483A', 
                # 'N439K', 'N501Y', 'R346K','A435S', 'G476S', 'F342L', 'N354D','W436R','A475V','K458R',
                # 'A435G','S373P','Y18A', 'N234A', 'S191A', 'Y505H','T478K','T95I','T478I', 'D405A',
                # 'H655Y', 'V26A', 'S375F','N226A','N236A','L212I','P23A','T376A','Q493K','K417N','F486A',
                # 'R491A','T547K','K386A','G446S','D24A','E188A','Q954H','R408S','N969K','A67V','N856K',
                # 'G339D','S371L','D20A','D405N','A372T','P691H','D769Y','L981F','N222Q','R235A','Q498R',
                # 'G446V','G142D','S28A','G496S','E309D','N679K','N764K','S371F',
                ]

dict_name_norm = {
                    'SARS':'SARS-CoV1',
                    'SARS-1':'SARS-CoV1',
                    'SARS-CoV1_WT':'SARS-CoV1',
                    'SARS-CoV':'SARS-CoV1', 
                    'SARS-2':'SARS-CoV2_WT',
                    'Original':'SARS-CoV2_WT',
                    'SARS-CoV2':'SARS-CoV2_WT',
                    'B.1.1.7':'Alpha', 
                    'B.1.617.2':'Delta',
                    'B.1.351':'Beta', 
                    'P.1':'Gamma',
                    'beta':'Beta', 
                    'lamnda':'Lamnda', 
                    'BA.2':'Omicron',
                    'delta':'Delta',
                    'gamma':'Gamma', 
                    'alpha':'Alpha', 
                    'epsilon':'Epsilon', 
                    'B.1.617.1':'Delta',
                    'MERS-CoV_WT':'MERS-CoV', 
                    'BA.1':'Omicron',
                    'WIV-1':'WIV1',
                    'Eplison':'Epsilon',
}


def get_antigen(_name):
    fp = open('./user_data/data_s2/%s.fasta' % _name)
    antigen_seq = ''
    for lne in fp:
        if not lne.startswith('>'):
            antigen_seq += lne.strip('\n')
    fp.close()
    return antigen_seq


def get_antigen_single(_name, item):
    fp = open('./user_data/data_s2/%s.fasta' % _name)
    antigen_seq_list = []
    for lne in fp:
        if not lne.startswith('>'):
            lne.strip('\n')
            for l in lne:
                antigen_seq_list.append(l)
            # antigen_seq += lne.strip('\n')
    fp.close()
    temp_s1 = item[:1]
    temp_s2 = item[-1:]
    loc_id = int(item[1:-1])

    # ss1 = antigen_seq_list[loc_id-2]
    # ss2 = antigen_seq_list[loc_id-1]
    # ss3 = antigen_seq_list[loc_id]
    # ss4 = antigen_seq_list[loc_id+1]
    antigen_seq_list[loc_id-1] = temp_s2

    antigen_seq = ''
    for ss in antigen_seq_list:
        antigen_seq += str(ss)
    
    return antigen_seq


def test_pipeline():
    try:
        test_root_dir = './tcdata/'
        test_aff_dir = test_root_dir + 'Affinity_test.csv'
        test_neu_dir = test_root_dir + 'Neutralization_test.csv'
        data_pd = pd.read_csv(test_aff_dir, encoding='gbk')
    except Exception as e:
        test_root_dir = './user_data/data_s2/'
        test_aff_dir = test_root_dir + './Affinity_train.csv'
        test_neu_dir = test_root_dir + './Neutralization_train.csv'
        data_pd = pd.read_csv(test_aff_dir, encoding='gbk')

    ## antigen
    dict_antigen_seqs = {}
    for item in antigen_list:
        try:
            seq = get_antigen(item)
        except Exception as e:
            try:
                item_name_norm = dict_name_norm[item]
                seq = get_antigen(item_name_norm)
            except Exception as e:
                print(e)
                # item_name_norm = 'SARS-CoV2_WT'
                # seq = get_antigen_single(item_name_norm, item)
                pass
        
        dict_antigen_seqs[item] = seq



    
    
    model_aff = GCN(128, [128,256,512,512], 0.5, 5)
    model_aff.load_state_dict(torch.load('./user_data/model_data_s3/model_aff_gcn_best.pth', map_location='cpu'), strict=True)
    model_aff = model_aff.to(device)
    model_aff.eval()

    list_results_a = predict_seqs_aff(data_pd, dict_antigen_seqs, model_aff)

    count = Counter(list_results_a)
    print(count)



    data_pd = pd.read_csv(test_neu_dir, encoding='gbk')

    model_neu = GCN(128, [128,256,512,512], 0.5, 5)
    # weights_gcn\model_neu_gcn_best.pth
    model_neu.load_state_dict(torch.load('./user_data/model_data_s3/model_neu_gcn_best.pth', map_location='cpu'), strict=True)
    model_neu = model_neu.to(device)
    model_neu.eval()

    list_results_n = predict_seqs_neu(data_pd, dict_antigen_seqs, model_neu)

    count = Counter(list_results_n)
    print(count)

    save_result(list_results_a, list_results_n)





if __name__ == '__main__':
    test_pipeline()