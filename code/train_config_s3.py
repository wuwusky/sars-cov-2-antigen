import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.optim as optim
from scipy import stats
import numpy as np
from tqdm import tqdm
import pandas as pd
import  ast



from test_feature_extract import *
from data_gen_s2 import *
# from model_resnet import *
from model_G import *


max_len = 128
max_len_ag = 1024
tokenizer_sim = Tokenizer()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# args, state_dict, forward_config = get_args_test()
logging.info(f'constructing omegafold')
model_omega = of.OmegaFold(of.make_config())
if "model" in state_dict:
    state_dict = state_dict.pop("model")
model_omega.load_state_dict(state_dict)
model_omega.eval()
model_omega.to(args.device)




def convert_onehot(label):
    label_np = np.array([0,0,0,0,0])
    for i in range(label):
        label_np[i] = 1
    # label_np[label] = 1
    return label_np


## [nan, nan, 'ARDNNYRNYYYYMDV', 'QQYGSSPPLT', 'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPGSASSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDPPEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQGSGYIPEAPRDGQAYVRKDGEWVLLSTFLENLYFQGDYKDDDDKHHHHHHHHH']

def split_seq(data_seq):
    temp_data_seq = data_seq[1:-2]
    temp_data_seq_list = temp_data_seq.split(', ')
    list_seqs = []
    for temp_seq in temp_data_seq_list:
        if temp_seq != 'nan':
            temp_seq = temp_seq[1:-1]
        else:
            temp_seq = ''
        list_seqs.append(temp_seq)

    return list_seqs
    pass



class dataset_anti(Dataset):
    def __init__(self, data_seq, data_lbl, status='train'):
        super(dataset_anti, self).__init__()
        self.data_seq = data_seq
        self.data_lbl = data_lbl
        self.status = status 

    def __getitem__(self, index):
        try:
            temp_data_seqs = ast.literal_eval(self.data_seq[index])
        except Exception as e:
            temp_data_seqs = split_seq(self.data_seq[index])
        temp_data_lbl = self.data_lbl[index]

        for n, temp_data_seq in enumerate(temp_data_seqs):
            if 'I253CCCC' in temp_data_seq:
                if '+I253CCCC' in temp_data_seq:
                    temp_data_seqs[n] = temp_data_seq.split('+')[0]
                else:
                    temp_data_seqs[n] = temp_data_seq[:-len('I253CCCC')]

        temp_seq_vh = temp_data_seqs[0]
        temp_seq_vl = temp_data_seqs[1]
        temp_seq_ch = temp_data_seqs[2]
        temp_seq_cl = temp_data_seqs[3]
        temp_seq_ag = temp_data_seqs[4]

        try:
            seq_vh = tokenizer_sim.gen_token_ids(temp_seq_vh)
            seq_vh_sp = tokenizer_sim.gen_token_ids_sp(temp_seq_vh)
        except Exception as e:
            seq_vh = [0]*10
        try:
            seq_vl = tokenizer_sim.gen_token_ids(temp_seq_vl)
            seq_vl_sp = tokenizer_sim.gen_token_ids_sp(temp_seq_vl)
        except Exception as e:
            seq_vl = [0]*10
        try:
            seq_ch = tokenizer_sim.gen_token_ids(temp_seq_ch)
            seq_ch_sp = tokenizer_sim.gen_token_ids_sp(temp_seq_ch)
        except Exception as e:
            seq_ch = [0]*10
        try:
            seq_cl = tokenizer_sim.gen_token_ids(temp_seq_cl)
            seq_cl_sp = tokenizer_sim.gen_token_ids_sp(temp_seq_cl)
        except Exception as e:
            seq_cl = [0]*10
        seq_ag = tokenizer_sim.gen_token_ids(temp_seq_ag)
        seq_ag_sp = tokenizer_sim.gen_token_ids_sp(temp_seq_ag)


        seq_vh = pad_to_max_seq(seq_vh, max_len, pad_id=0)
        seq_vl = pad_to_max_seq(seq_vl, max_len, pad_id=0)
        seq_ch = pad_to_max_seq(seq_ch, max_len, pad_id=0)
        seq_cl = pad_to_max_seq(seq_cl, max_len, pad_id=0)
        seq_ag = pad_to_max_seq(seq_ag, 1500, pad_id=0)

        seq_vh_sp = pad_to_max_seq(seq_vh_sp, max_len, pad_id=0)
        seq_vl_sp = pad_to_max_seq(seq_vl_sp, max_len, pad_id=0)
        seq_ch_sp = pad_to_max_seq(seq_ch_sp, max_len, pad_id=0)
        seq_cl_sp = pad_to_max_seq(seq_cl_sp, max_len, pad_id=0)
        seq_ag_sp = pad_to_max_seq(seq_ag_sp, 1500, pad_id=0)

        # if self.status == 'train':
        #     if random.random() < 0.1:
        #         seq_vh = np.zeros_like(seq_vh)
        #     if random.random() < 0.1:
        #         seq_vl = np.zeros_like(seq_vl)
        #     # if random.random() < 0.1:
            #     seq_ch = np.zeros_like(seq_ch)
            # if random.random() < 0.1:
            #     seq_cl = np.zeros_like(seq_cl)

        seq_chs = np.stack([seq_vh, seq_vl, seq_ch, seq_cl, seq_vh_sp, seq_vl_sp, seq_ch_sp, seq_cl_sp], axis=1)
        seq_chs_angen = np.stack([seq_ag, seq_ag_sp], axis=1)

        if self.status == 'train':
            seq_chs = random_seq_mask(seq_chs)

        data_seq_tensor = torch.from_numpy(seq_chs).permute(1,0)[:,:]
        data_seq_angen_tensor = torch.from_numpy(seq_chs_angen).permute(1,0)[:,:]

        data_lbl_tensor = torch.from_numpy(np.array(temp_data_lbl))
        data_lbl_tensor_onehot = torch.from_numpy(convert_onehot(temp_data_lbl))

        return data_seq_tensor.float(), data_seq_angen_tensor.float(), \
                data_lbl_tensor.float(), data_lbl_tensor_onehot.float()
    
    def __len__(self):
        return len(self.data_seq)



class dataset_anti_graph(Dataset):
    def __init__(self, data_seq, data_lbl, status='train'):
        super(dataset_anti_graph, self).__init__()
        self.data_seq = data_seq
        self.data_lbl = data_lbl
        self.status = status 

    def __getitem__(self, index):
        try:
            temp_data_seqs = ast.literal_eval(self.data_seq[index])
        except Exception as e:
            temp_data_seqs = split_seq(self.data_seq[index])
        temp_data_lbl = self.data_lbl[index]

        for n, temp_data_seq in enumerate(temp_data_seqs):
            if 'I253' in temp_data_seq:
                if '+I253' in temp_data_seq:
                    temp_data_seqs[n] = temp_data_seq.split('+I253')[0]
                else:
                    temp_data_seqs[n] = temp_data_seq.split('I253')[0]
            elif ' ' in temp_data_seq:
                temp_data_seqs[n] = temp_data_seq.split(' ')[0]

        temp_seq_vh = temp_data_seqs[0]
        temp_seq_vl = temp_data_seqs[1]
        temp_seq_ch = temp_data_seqs[2]
        temp_seq_cl = temp_data_seqs[3]
        temp_seq_ag = temp_data_seqs[4]


        
        with torch.no_grad():
            vh = data_process_sim(temp_seq_vh, max_len)
            features = model_omega.feature_extract(vh, forward_config)
            n_h, e_h = features

            vl = data_process_sim(temp_seq_vl, max_len)
            features = model_omega.feature_extract(vl, forward_config)
            n_l, e_l = features

            ag = data_process_sim(temp_seq_ag, max_len_ag)
            features = model_omega.feature_extract(ag, forward_config)
            n_g, e_g = features


        data_lbl_tensor = torch.from_numpy(np.array(temp_data_lbl))
        data_lbl_tensor_onehot = torch.from_numpy(convert_onehot(temp_data_lbl))

        return n_h, e_h, n_l, e_l, n_g, e_g, \
            data_lbl_tensor, data_lbl_tensor_onehot
    
    def __len__(self):
        return len(self.data_seq)



class dataset_anti_graph_pre(Dataset):
    def __init__(self, data_seq, data_lbl, status='train'):
        super(dataset_anti_graph_pre, self).__init__()
        self.data_seq = data_seq
        self.data_lbl = data_lbl
        self.status = status 

    def __getitem__(self, index):
        n_h, e_h, n_l, e_l, n_g, e_g = self.data_seq[index]
        temp_data_lbl = self.data_lbl[index]


        data_lbl_tensor = torch.from_numpy(np.array(temp_data_lbl))
        data_lbl_tensor_onehot = torch.from_numpy(convert_onehot(temp_data_lbl))

        return n_h, e_h, n_l, e_l, n_g, e_g, \
            data_lbl_tensor, data_lbl_tensor_onehot
    
    def __len__(self):
        return len(self.data_seq)




class dataset_anti_mt(Dataset):
    def __init__(self, data_seq, data_lbl, status='train'):
        super(dataset_anti_mt, self).__init__()
        self.data_seq = data_seq
        self.data_lbl = data_lbl
        self.status = status 

    def __getitem__(self, index):
        try:
            temp_data_seqs = ast.literal_eval(self.data_seq[index])
        except Exception as e:
            temp_data_seqs = split_seq(self.data_seq[index])
        temp_data_lbl = self.data_lbl[index]

        temp_seq_vh = temp_data_seqs[0]
        temp_seq_vl = temp_data_seqs[1]
        temp_seq_ch = temp_data_seqs[2]
        temp_seq_cl = temp_data_seqs[3]
        temp_seq_ag = temp_data_seqs[4]

        loc_cdrh = [temp_seq_vh.find(temp_seq_ch)/128, len(temp_seq_ch)/48]
        loc_cdrl = [temp_seq_vl.find(temp_seq_cl)/128, len(temp_seq_cl)/48]

        try:
            seq_vh = tokenizer_sim.gen_token_ids(temp_seq_vh)
            seq_vh_sp = tokenizer_sim.gen_token_ids_sp(temp_seq_vh)
        except Exception as e:
            seq_vh = [0]*10
        try:
            seq_vl = tokenizer_sim.gen_token_ids(temp_seq_vl)
            seq_vl_sp = tokenizer_sim.gen_token_ids_sp(temp_seq_vl)
        except Exception as e:
            seq_vl = [0]*10
        try:
            seq_ch = tokenizer_sim.gen_token_ids(temp_seq_ch)
            seq_ch_sp = tokenizer_sim.gen_token_ids_sp(temp_seq_ch)
        except Exception as e:
            seq_ch = [0]*10
        try:
            seq_cl = tokenizer_sim.gen_token_ids(temp_seq_cl)
            seq_cl_sp = tokenizer_sim.gen_token_ids_sp(temp_seq_cl)
        except Exception as e:
            seq_cl = [0]*10
        seq_ag = tokenizer_sim.gen_token_ids(temp_seq_ag)
        seq_ag_sp = tokenizer_sim.gen_token_ids_sp(temp_seq_ag)

        
        
        seq_vh = pad_to_max_seq(seq_vh, max_len, pad_id=0)
        seq_vl = pad_to_max_seq(seq_vl, max_len, pad_id=0)
        seq_ch = pad_to_max_seq(seq_ch, max_len, pad_id=0)
        seq_cl = pad_to_max_seq(seq_cl, max_len, pad_id=0)
        seq_ag = pad_to_max_seq(seq_ag, 1500, pad_id=0)

        seq_vh_sp = pad_to_max_seq(seq_vh_sp, max_len, pad_id=0)
        seq_vl_sp = pad_to_max_seq(seq_vl_sp, max_len, pad_id=0)
        seq_ch_sp = pad_to_max_seq(seq_ch_sp, max_len, pad_id=0)
        seq_cl_sp = pad_to_max_seq(seq_cl_sp, max_len, pad_id=0)
        seq_ag_sp = pad_to_max_seq(seq_ag_sp, 1500, pad_id=0)

        # if self.status == 'train':
        #     if random.random() < 0.1:
        #         seq_vh = np.zeros_like(seq_vh)
        #     if random.random() < 0.1:
        #         seq_vl = np.zeros_like(seq_vl)
        #     # if random.random() < 0.1:
            #     seq_ch = np.zeros_like(seq_ch)
            # if random.random() < 0.1:
            #     seq_cl = np.zeros_like(seq_cl)
        

        

        seq_chs = np.stack([seq_vh, seq_vl, seq_ch, seq_cl, seq_vh_sp, seq_vl_sp, seq_ch_sp, seq_cl_sp], axis=1)
        seq_chs_angen = np.stack([seq_ag, seq_ag_sp], axis=1)

        if self.status == 'train':
            seq_chs = random_seq_mask(seq_chs)

        data_seq_tensor = torch.from_numpy(seq_chs).permute(1,0)[:,:]
        data_seq_angen_tensor = torch.from_numpy(seq_chs_angen).permute(1,0)[:,:]

        data_lbl_tensor = torch.from_numpy(np.array(temp_data_lbl))
        data_lbl_tensor_onehot = torch.from_numpy(convert_onehot(temp_data_lbl))

        if self.status == 'train':
            data_ch_loc = torch.from_numpy(np.array(loc_cdrh))
            data_cl_loc = torch.from_numpy(np.array(loc_cdrl))

            return data_seq_tensor.float(), data_seq_angen_tensor.float(), \
                    data_lbl_tensor.float(), data_lbl_tensor_onehot.float(), \
                    data_ch_loc.float(), data_cl_loc.float()

        else:
            return data_seq_tensor.float(), data_seq_angen_tensor.float(), \
                    data_lbl_tensor.float(), data_lbl_tensor_onehot.float() \

    
    def __len__(self):
        return len(self.data_seq)


class dataset_anti_mt_str(Dataset):
    def __init__(self, data_seq, data_lbl, data_str, status='train'):
        super(dataset_anti_mt_str, self).__init__()
        self.data_seq = data_seq
        self.data_lbl = data_lbl
        self.status = status
        self.data_str = data_str

    def __getitem__(self, index):
        try:
            temp_data_seqs = ast.literal_eval(self.data_seq[index])
        except Exception as e:
            temp_data_seqs = split_seq(self.data_seq[index])
        temp_data_lbl = self.data_lbl[index]

        temp_seq_vh = temp_data_seqs[0]
        temp_seq_vl = temp_data_seqs[1]
        temp_seq_ch = temp_data_seqs[2]
        temp_seq_cl = temp_data_seqs[3]
        temp_seq_ag = temp_data_seqs[4]

        loc_cdrh = [temp_seq_vh.find(temp_seq_ch)/max_len, (temp_seq_vh.find(temp_seq_ch) + len(temp_seq_ch))/max_len]
        loc_cdrl = [temp_seq_vl.find(temp_seq_cl)/max_len, (temp_seq_vl.find(temp_seq_cl) + len(temp_seq_cl))/max_len]

        try:
            seq_vh = tokenizer_sim.gen_token_ids(temp_seq_vh)
            seq_vh_sp = tokenizer_sim.gen_token_ids_sp(temp_seq_vh)
        except Exception as e:
            seq_vh = [0]*10
        try:
            seq_vl = tokenizer_sim.gen_token_ids(temp_seq_vl)
            seq_vl_sp = tokenizer_sim.gen_token_ids_sp(temp_seq_vl)
        except Exception as e:
            seq_vl = [0]*10
        try:
            seq_ch = tokenizer_sim.gen_token_ids(temp_seq_ch)
            seq_ch_sp = tokenizer_sim.gen_token_ids_sp(temp_seq_ch)
        except Exception as e:
            seq_ch = [0]*10
        try:
            seq_cl = tokenizer_sim.gen_token_ids(temp_seq_cl)
            seq_cl_sp = tokenizer_sim.gen_token_ids_sp(temp_seq_cl)
        except Exception as e:
            seq_cl = [0]*10
        seq_ag = tokenizer_sim.gen_token_ids(temp_seq_ag)
        seq_ag_sp = tokenizer_sim.gen_token_ids_sp(temp_seq_ag)

        
        
        seq_vh = pad_to_max_seq(seq_vh, max_len, pad_id=0)
        seq_vl = pad_to_max_seq(seq_vl, max_len, pad_id=0)
        seq_ch = pad_to_max_seq(seq_ch, max_len, pad_id=0)
        seq_cl = pad_to_max_seq(seq_cl, max_len, pad_id=0)
        seq_ag = pad_to_max_seq(seq_ag, 1500, pad_id=0)

        seq_vh_sp = pad_to_max_seq(seq_vh_sp, max_len, pad_id=0)
        seq_vl_sp = pad_to_max_seq(seq_vl_sp, max_len, pad_id=0)
        seq_ch_sp = pad_to_max_seq(seq_ch_sp, max_len, pad_id=0)
        seq_cl_sp = pad_to_max_seq(seq_cl_sp, max_len, pad_id=0)
        seq_ag_sp = pad_to_max_seq(seq_ag_sp, 1500, pad_id=0)

        # if self.status == 'train':
        #     if random.random() < 0.1:
        #         seq_vh = np.zeros_like(seq_vh)
        #     if random.random() < 0.1:
        #         seq_vl = np.zeros_like(seq_vl)
        #     # if random.random() < 0.1:
            #     seq_ch = np.zeros_like(seq_ch)
            # if random.random() < 0.1:
            #     seq_cl = np.zeros_like(seq_cl)
        
        ### pdb structure data
        temp_data = self.data_str[index % len(self.data_str)]
        temp_map = temp_data[0]
        temp_seq = temp_data[1]

        map_seq = tokenizer_sim.gen_token_ids(temp_seq)
        map_seq = pad_to_max_seq(map_seq, 256, pad_id=0)
        map_len = temp_map.shape[-1]
        if map_len >= 256:
            map_str = temp_map[:256, :256]
        else:
            pad_size = 256-map_len
            map_str = np.pad(temp_map, ((0, pad_size),(0, pad_size)))
        # map_str[map_str>10] = 0
        # map_str[map_str>0] = 1
        
        map_seq_tensor = torch.from_numpy(np.stack([map_seq], axis=0))
        map_str_tensor = torch.from_numpy(np.stack([map_str], axis=0))
        

        seq_chs = np.stack([seq_vh, seq_vl, seq_ch, seq_cl, seq_vh_sp, seq_vl_sp, seq_ch_sp, seq_cl_sp], axis=1)
        seq_chs_angen = np.stack([seq_ag, seq_ag_sp], axis=1)

        # if self.status == 'train':
        #     seq_chs = random_seq_mask(seq_chs)

        data_seq_tensor = torch.from_numpy(seq_chs).permute(1,0)[:,:]
        data_seq_angen_tensor = torch.from_numpy(seq_chs_angen).permute(1,0)[:,:]

        data_lbl_tensor = torch.from_numpy(np.array(temp_data_lbl))
        data_lbl_tensor_onehot = torch.from_numpy(convert_onehot(temp_data_lbl))

        if self.status == 'train':
            data_ch_loc = torch.from_numpy(np.array(loc_cdrh))
            data_cl_loc = torch.from_numpy(np.array(loc_cdrl))

            return data_seq_tensor.float(), data_seq_angen_tensor.float(), \
                    data_lbl_tensor.float(), data_lbl_tensor_onehot.float(), \
                    data_ch_loc.float(), data_cl_loc.float(), \
                    map_seq_tensor.float(), map_str_tensor.float(),

        else:
            return data_seq_tensor.float(), data_seq_angen_tensor.float(), \
                    data_lbl_tensor.float(), data_lbl_tensor_onehot.float() \

    
    def __len__(self):
        return len(self.data_seq)



from sklearn.metrics import mean_squared_error
# def rmse(y_true, y_pred):
#     return np.sqrt(mean_squared_error(y_true, y_pred))
def rmse(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.sqrt(np.square(np.subtract(actual, pred)).mean())


batch_size = 5
num_w = 0
learn_rate = 1e-4
max_epoch = 30


model = GCN(128, [128,256,512,512], 0.5, 5).to(device)
model_eval = GCN(128, [128,256,512,512], 0.5, 5).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learn_rate, weight_decay=5e-4)
lr_sh = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25])

# loss_fun = nn.CrossEntropyLoss()
# loss_fun = nn.MSELoss()
loss_fun = nn.SmoothL1Loss()

loss_fun_str = nn.SmoothL1Loss()
# loss_fun = nn.BCELoss()

model_save_dir = './weights_gcn/'
os.makedirs(model_save_dir, exist_ok=True)