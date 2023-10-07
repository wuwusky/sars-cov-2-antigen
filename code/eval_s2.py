import pandas as pd
import  torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from data_gen_s2 import *
from model_resnet_s2 import *
max_len = 128
tokenizer_sim = Tokenizer()
# from tqdm import tqdm
from tqdm.contrib import tzip
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



def ex_seq_info_N(pd_info, dict_antigen_seqs):
    seq_feature_names = ['VH or VHH','VL','CDRH3','CDRL3', 'Neutralising Vs']
    seq_infos_np = pd_info[seq_feature_names].values

    list_data_seqs = []
    list_data_angen = []
    for seq_info in seq_infos_np:
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

        seq_vh_sp = tokenizer_sim.gen_token_ids_sp(seq_vh)
        seq_vl_sp = tokenizer_sim.gen_token_ids_sp(seq_vl)
        seq_ch_sp = tokenizer_sim.gen_token_ids_sp(seq_cdrh)
        seq_cl_sp = tokenizer_sim.gen_token_ids_sp(seq_cdrl)
        seq_ag_sp = tokenizer_sim.gen_token_ids_sp(seq_angen)

        seq_vh = tokenizer_sim.gen_token_ids(seq_vh)
        seq_vl = tokenizer_sim.gen_token_ids(seq_vl)
        seq_ch = tokenizer_sim.gen_token_ids(seq_cdrh)
        seq_cl = tokenizer_sim.gen_token_ids(seq_cdrl)
        seq_ag = tokenizer_sim.gen_token_ids(seq_angen)

        seq_vh_sp = pad_to_max_seq(seq_vh_sp, max_len, pad_id=0)
        seq_vl_sp = pad_to_max_seq(seq_vl_sp, max_len, pad_id=0)
        seq_ch_sp = pad_to_max_seq(seq_ch_sp, max_len, pad_id=0)
        seq_cl_sp = pad_to_max_seq(seq_cl_sp, max_len, pad_id=0)
        seq_ag_sp = pad_to_max_seq(seq_ag_sp, 1500, pad_id=0)

        seq_vh = pad_to_max_seq(seq_vh, max_len, pad_id=0)
        seq_vl = pad_to_max_seq(seq_vl, max_len, pad_id=0)
        seq_ch = pad_to_max_seq(seq_ch, max_len, pad_id=0)
        seq_cl = pad_to_max_seq(seq_cl, max_len, pad_id=0)
        seq_ag = pad_to_max_seq(seq_ag, 1500, pad_id=0)



        seq_chs = np.stack([seq_vh, seq_vl, seq_ch, seq_cl, seq_vh_sp, seq_vl_sp, seq_ch_sp, seq_cl_sp], axis=1)

        list_data_seqs.append(seq_chs)
        list_data_angen.append(np.stack([seq_ag, seq_ag_sp], axis=1))

    data_seqs_np = np.array(list_data_seqs)
    data_angen_np = np.array(list_data_angen)
    return data_seqs_np, data_angen_np



def ex_seq_info_A(pd_info, dict_antigen_seqs):
    seq_feature_names = ['VH or VHH','VL','CDRH3','CDRL3', 'Binds to']
    seq_infos_np = pd_info[seq_feature_names].values

    list_data_seqs = []
    list_data_angen = []
    for seq_info in seq_infos_np:
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

        seq_vh_sp = tokenizer_sim.gen_token_ids_sp(seq_vh)
        seq_vl_sp = tokenizer_sim.gen_token_ids_sp(seq_vl)
        seq_ch_sp = tokenizer_sim.gen_token_ids_sp(seq_cdrh)
        seq_cl_sp = tokenizer_sim.gen_token_ids_sp(seq_cdrl)
        seq_ag_sp = tokenizer_sim.gen_token_ids_sp(seq_angen)

        seq_vh = tokenizer_sim.gen_token_ids(seq_vh)
        seq_vl = tokenizer_sim.gen_token_ids(seq_vl)
        seq_ch = tokenizer_sim.gen_token_ids(seq_cdrh)
        seq_cl = tokenizer_sim.gen_token_ids(seq_cdrl)
        seq_ag = tokenizer_sim.gen_token_ids(seq_angen)

        seq_vh_sp = pad_to_max_seq(seq_vh_sp, max_len, pad_id=0)
        seq_vl_sp = pad_to_max_seq(seq_vl_sp, max_len, pad_id=0)
        seq_ch_sp = pad_to_max_seq(seq_ch_sp, max_len, pad_id=0)
        seq_cl_sp = pad_to_max_seq(seq_cl_sp, max_len, pad_id=0)
        seq_ag_sp = pad_to_max_seq(seq_ag_sp, 1500, pad_id=0)

        seq_vh = pad_to_max_seq(seq_vh, max_len, pad_id=0)
        seq_vl = pad_to_max_seq(seq_vl, max_len, pad_id=0)
        seq_ch = pad_to_max_seq(seq_ch, max_len, pad_id=0)
        seq_cl = pad_to_max_seq(seq_cl, max_len, pad_id=0)
        seq_ag = pad_to_max_seq(seq_ag, 1500, pad_id=0)

        
        seq_chs = np.stack([seq_vh, seq_vl, seq_ch, seq_cl, seq_vh_sp, seq_vl_sp, seq_ch_sp, seq_cl_sp], axis=1)

        list_data_seqs.append(seq_chs)
        list_data_angen.append(np.stack([seq_ag, seq_ag_sp], axis=1))

    data_seqs_np = np.array(list_data_seqs)
    data_angen_np = np.array(list_data_angen)
    return data_seqs_np, data_angen_np



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
        test_root_dir = './data_s2/'
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
                # print(e)
                # item_name_norm = 'SARS-CoV2_WT'
                # seq = get_antigen_single(item_name_norm, item)
                pass
        
        dict_antigen_seqs[item] = seq



    data_seqs_np, data_angen_np = ex_seq_info_A(data_pd, dict_antigen_seqs)
    
    
    model_aff = resnet18_mt()
    model_aff.load_state_dict(torch.load('./user_data/model_data_s2/model_aff_mt_str_best.pth', map_location='cpu'), strict=True)
    model_aff = model_aff.to(device)
    # model_aff.eval()

    list_results_a = []
    for  temp_data_seq, temp_antigen_seq in tzip(data_seqs_np, data_angen_np, ncols=100):
        data_seq_tensor = torch.from_numpy(temp_data_seq).permute(1,0).float()
        data_seq_angen_tensor = torch.from_numpy(temp_antigen_seq).permute(1,0).float()

        data_seq_tensor = torch.stack([data_seq_tensor], dim=0).to(device)
        data_seq_angen_tensor = torch.stack([data_seq_angen_tensor], dim=0).to(device)

        with torch.no_grad():
            out = model_aff.forward(data_seq_tensor, data_seq_angen_tensor)

        temp_preds = torch.sum(out.detach().cpu(), dim=1).view(-1).numpy()
        list_results_a += (temp_preds+0.5).astype(np.int8).tolist()

    count = Counter(list_results_a)
    print(count)



    data_pd = pd.read_csv(test_neu_dir, encoding='gbk')
    data_seqs_np, data_angen_np = ex_seq_info_N(data_pd, dict_antigen_seqs)

    model_neu = resnet18_mt()
    model_neu.load_state_dict(torch.load('/user_data/model_data_s2/model_neu_mt_str_best.pth', map_location='cpu'), strict=True)
    model_neu = model_neu.to(device)
    # model_neu.eval()

    list_results_n = []
    for  temp_data_seq, temp_antigen_seq in tzip(data_seqs_np, data_angen_np, ncols=100):
        data_seq_tensor = torch.from_numpy(temp_data_seq).permute(1,0).float()
        data_seq_angen_tensor = torch.from_numpy(temp_antigen_seq).permute(1,0).float()

        data_seq_tensor = torch.stack([data_seq_tensor], dim=0).to(device)
        data_seq_angen_tensor = torch.stack([data_seq_angen_tensor], dim=0).to(device)

        with torch.no_grad():
            out = model_neu.forward(data_seq_tensor, data_seq_angen_tensor)

        temp_preds = torch.sum(out.detach().cpu(), dim=1).view(-1).numpy()
        list_results_n += (temp_preds+0.5).astype(np.int8).tolist()
    count = Counter(list_results_n)
    print(count)

    save_result(list_results_a, list_results_n)





if __name__ == '__main__':
    test_pipeline()