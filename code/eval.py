import pandas as pd
import  torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from model_s1 import *
from data_gen import *
from model_resnet_s1 import *
max_len = 128
tokenizer_sim = Tokenizer()
from tqdm import tqdm

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
    
    if os.path.exists('./prediction_result/') is False:
        os.makedirs('./prediction_result/')
    
    with open('./prediction_result/result.csv', mode='w', encoding='gb2312', newline='') as f:
        for row in list_lines:
            f.write(row+'\n')
    print('test')


def load_test(data_dir):
    with open(data_dir, mode='r', encoding='gb2312') as f:
        pd_info = pd.read_csv(f)
    temp_names_np = pd_info.values[:,0]
    return temp_names_np.tolist()


def ex_seq_info(pd_info):
    seq_feature_names = ['VH or VHH','VL','CDRH3','CDRL3']
    seq_infos_np = pd_info[seq_feature_names].values

    list_data_seqs = []
    for seq_info in seq_infos_np:
        seq_vh, seq_vl, seq_cdrh, seq_cdrl = seq_info

        seq_vh = str(seq_vh)
        seq_vl = str(seq_vl)
        seq_cdrh = str(seq_cdrh)
        seq_cdrl = str(seq_cdrl)

        seq_vh_ids = tokenizer_sim.gen_token_ids(seq_vh)
        seq_vl_ids = tokenizer_sim.gen_token_ids(seq_vl)
        seq_cdrh_ids = tokenizer_sim.gen_token_ids(seq_cdrh)
        seq_cdrl_ids = tokenizer_sim.gen_token_ids(seq_cdrl)

        seq_vh_pad = pad_to_max_seq(seq_vh_ids, max_len, pad_id=0)
        seq_vl_pad = pad_to_max_seq(seq_vl_ids, max_len, pad_id=0)
        seq_cdrh_pad = pad_to_max_seq(seq_cdrh_ids, max_len, pad_id=0)
        seq_cdrl_pad = pad_to_max_seq(seq_cdrl_ids, max_len, pad_id=0)

        seq_chs = np.stack([seq_vh_pad, seq_vl_pad, seq_cdrh_pad, seq_cdrl_pad], axis=1)

        list_data_seqs.append(seq_chs)

    data_seqs_np = np.array(list_data_seqs)
    return data_seqs_np



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

        seq_vh_ids = tokenizer_sim.gen_token_ids(seq_vh)
        seq_vl_ids = tokenizer_sim.gen_token_ids(seq_vl)
        seq_cdrh_ids = tokenizer_sim.gen_token_ids(seq_cdrh)
        seq_cdrl_ids = tokenizer_sim.gen_token_ids(seq_cdrl)
        seq_angen_ids = tokenizer_sim.gen_token_ids(seq_angen)

        seq_vh_pad = pad_to_max_seq(seq_vh_ids, max_len, pad_id=0)
        seq_vl_pad = pad_to_max_seq(seq_vl_ids, max_len, pad_id=0)
        seq_cdrh_pad = pad_to_max_seq(seq_cdrh_ids, max_len, pad_id=0)
        seq_cdrl_pad = pad_to_max_seq(seq_cdrl_ids, max_len, pad_id=0)
        seq_angen_pad = pad_to_max_seq(seq_angen_ids, 1500, pad_id=0)

        seq_chs = np.stack([seq_vh_pad, seq_vl_pad, seq_cdrh_pad, seq_cdrl_pad], axis=1)

        list_data_seqs.append(seq_chs)
        list_data_angen.append(np.stack([seq_angen_pad], axis=1))

    data_seqs_np = np.array(list_data_seqs)
    data_angen_np = np.array(list_data_angen)
    return data_seqs_np, data_angen_np


def test_pipeline():
    test_root_dir = './tcdata/'
    test_aff_dir = test_root_dir + 'Affinity_test.csv'
    test_neu_dir = test_root_dir + 'Neutralization_test.csv'

    # test_root_dir = './data/'
    # test_aff_dir = test_root_dir + './Affinity_train.csv'
    # test_neu_dir = test_root_dir + './Neutralization_train.csv'

    ## affi
    data_pd = pd.read_csv(test_aff_dir, encoding='gbk')
    data_seqs_np = ex_seq_info(data_pd)
    with open('../user_data/data/SARS-CoV2_WT.fasta') as f:
        antigen_seq = []
        for line in f:
            if line.startswith('>'):
                continue
            else:
                antigen_seq.append(line.strip('\n'))
    antigen_seq = np.stack([pad_to_max_seq(tokenizer_sim.gen_token_ids(antigen_seq), 1280, pad_id=0)], axis=1)
    
    
    model_aff = resnet50()
    model_aff.load_state_dict(torch.load('../user_data/weights/model_aff_best.pth', map_location='cpu'), strict=False)
    model_aff = model_aff.to(device)
    model_aff.eval()

    list_results_a = []
    for temp_data_seq in tqdm(data_seqs_np, ncols=100):
        data_seq_tensor = torch.from_numpy(temp_data_seq).permute(1,0).float()
        data_seq_angen_tensor = torch.from_numpy(antigen_seq).permute(1,0).float()

        data_seq_tensor = torch.stack([data_seq_tensor], dim=0).to(device)
        data_seq_angen_tensor = torch.stack([data_seq_angen_tensor], dim=0).to(device)

        with torch.no_grad():
            out = model_aff.forward(data_seq_tensor, data_seq_angen_tensor)

        temp_preds = torch.sum(out.detach().cpu(), dim=1)
        list_results_a.append(int(temp_preds+0.5))



    ## neu
    antigen_type_list = ['SARS-CoV1', 'SARS-CoV2_WT', 
                    'SARS-CoV2_Alpha', 'SARS-CoV2_Beta', 'SARS-CoV2_Gamma',
                    'SARS-CoV2_Delta', 'SARS-CoV2_Kappa', 'SARS-CoV2_Omicron']

    dict_antigen_seqs = {}
    for temp_antigen_name in antigen_type_list:
        temp_antigen_dir = '../user_data/data/' + temp_antigen_name + '.fasta'
        try:
            with open(temp_antigen_dir) as f:
                antigen_seq = []
                for line in f:
                    if line.startswith('>'):
                        continue
                    else:
                        antigen_seq.append(line.strip('\n'))
        except Exception as e:
            temp_antigen_dir = '../user_data/data/' + temp_antigen_name.split('_')[-1] + '.fasta'
            with open(temp_antigen_dir) as f:
                antigen_seq = []
                for line in f:
                    if line.startswith('>'):
                        continue
                    else:
                        antigen_seq.append(line.strip('\n'))
        print(len(antigen_seq[0]))
        dict_antigen_seqs[temp_antigen_name] = antigen_seq[0]


    data_pd = pd.read_csv(test_neu_dir, encoding='gbk')
    data_seqs_np, data_angen_np = ex_seq_info_N(data_pd, dict_antigen_seqs)

    model_neu = resnet50()
    model_neu.load_state_dict(torch.load('../user_data/weights/model_neu_best.pth', map_location='cpu'), strict=False)
    model_neu = model_neu.to(device)
    model_neu.eval()

    list_results_n = []
    for  temp_data_seq, temp_antigen_seq in zip(data_seqs_np, data_angen_np):
        data_seq_tensor = torch.from_numpy(temp_data_seq).permute(1,0).float()
        data_seq_angen_tensor = torch.from_numpy(temp_antigen_seq).permute(1,0).float()

        data_seq_tensor = torch.stack([data_seq_tensor], dim=0).to(device)
        data_seq_angen_tensor = torch.stack([data_seq_angen_tensor], dim=0).to(device)

        with torch.no_grad():
            out = model_neu.forward(data_seq_tensor, data_seq_angen_tensor)

        temp_preds = torch.sum(out.detach().cpu(), dim=1)
        list_results_n.append(int(temp_preds+0.5))
        

    save_result(list_results_a, list_results_n)





if __name__ == '__main__':
    test_pipeline()