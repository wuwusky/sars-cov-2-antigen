import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.optim as optim
from scipy import stats
import numpy as np
from tqdm import tqdm
import pandas as pd
import  ast




from model_s1 import *
from data_gen import *
from model_resnet_s1 import *
max_len = 128
tokenizer_sim = Tokenizer()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def convert_onehot(label):
    label_np = np.array([0,0,0,0,0])
    for i in range(label):
        label_np[i] = 1
    # label_np[label] = 1
    return label_np

class dataset_anti(Dataset):
    def __init__(self, data_seq, data_lbl, status='train'):
        super(dataset_anti, self).__init__()
        self.data_seq = data_seq
        self.data_lbl = data_lbl
        self.status = status 

    def __getitem__(self, index):
        temp_data_seqs = ast.literal_eval(self.data_seq[index])
        temp_data_lbl = self.data_lbl[index]

        temp_seq_vh = temp_data_seqs[0]
        temp_seq_vl = temp_data_seqs[1]
        temp_seq_ch = temp_data_seqs[2]
        temp_seq_cl = temp_data_seqs[3]
        temp_seq_ag = temp_data_seqs[4]

        seq_vh = tokenizer_sim.gen_token_ids(temp_seq_vh)
        seq_vl = tokenizer_sim.gen_token_ids(temp_seq_vl)
        seq_ch = tokenizer_sim.gen_token_ids(temp_seq_ch)
        seq_cl = tokenizer_sim.gen_token_ids(temp_seq_cl)
        seq_ag = tokenizer_sim.gen_token_ids(temp_seq_ag)

        seq_vh = pad_to_max_seq(seq_vh, max_len, pad_id=0)
        seq_vl = pad_to_max_seq(seq_vl, max_len, pad_id=0)
        seq_ch = pad_to_max_seq(seq_ch, max_len, pad_id=0)
        seq_cl = pad_to_max_seq(seq_cl, max_len, pad_id=0)
        seq_ag = pad_to_max_seq(seq_ag, 1500, pad_id=0)

        if self.status == 'train':
            # if random.random() < 0.1:
            #     seq_vh = np.zeros_like(seq_vh)
            # if random.random() < 0.1:
            #     seq_vl = np.zeros_like(seq_vl)
            # if random.random() < 0.1:
            #     seq_ch = np.zeros_like(seq_ch)
            # if random.random() < 0.1:
            #     seq_cl = np.zeros_like(seq_cl)
            pass

        seq_chs = np.stack([seq_vh, seq_vl, seq_ch, seq_cl], axis=1)
        seq_chs_angen = np.stack([seq_ag], axis=1)

        data_seq_tensor = torch.from_numpy(seq_chs).permute(1,0)[:,:]
        data_seq_angen_tensor = torch.from_numpy(seq_chs_angen).permute(1,0)[:,:]

        data_lbl_tensor = torch.from_numpy(np.array(temp_data_lbl))
        data_lbl_tensor_onehot = torch.from_numpy(convert_onehot(temp_data_lbl))

        return data_seq_tensor.float(), data_seq_angen_tensor.float(), data_lbl_tensor.float(), data_lbl_tensor_onehot.float()
    
    def __len__(self):
        return len(self.data_seq)

from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

    



def main_train():
    batch_size = 32
    num_w = 0
    learn_rate = 1e-4
    max_epoch = 200
    num_sample = 100
    flag_sample = False


    
    data_dir = r'./user_data\data\train_data_full.csv'
    data_ex_dir = r'./user_data\data\train_data_full_ex.csv'

    data = pd.read_csv(data_dir)
    data_seq = data['Sequence'].values.tolist()
    data_lbl = data['Label'].values.tolist()

    # temp_data_seqs = ast.literal_eval(data_seq[0])
    # for ll in temp_data_seqs:
    #     print(len(ll))


    data_ex = pd.read_csv(data_ex_dir)
    data_seq_ex = data_ex['Sequence'].values.tolist()
    data_lbl_ex = data_ex['Label'].values.tolist()



    data_all = data_seq + data_seq_ex
    data_lbl = data_lbl + data_lbl_ex

    data_seq_train = data_all[:-num_sample]
    data_lbl_train = data_lbl[:-num_sample]
    data_seq_valid = data_all[-num_sample:]
    data_lbl_valid = data_lbl[-num_sample:]

    
    print('train:')
    dict_lbl2weight = {}
    for i in range(1,6,1):
        # print(i, data_lbl_train.count(i))
        temp_num = data_lbl_train.count(i)
        temp_ratio = 1-(temp_num/len(data_lbl_train))
        dict_lbl2weight[str(i)] = temp_ratio
        print(i, temp_num)
    print('valid:')
    for i in range(1,6,1):
        print(i, data_lbl_valid.count(i))

    label_weights_train = []
    for temp_lbl in data_lbl_train:
        label_weights_train.append(dict_lbl2weight[str(temp_lbl)])

    dataset_train = dataset_anti(data_seq_train, data_lbl_train, 'train')
    dataset_valid = dataset_anti(data_seq_valid, data_lbl_valid, 'valid')
    print('dataset init finished')


    if flag_sample:
        sampler_my = WeightedRandomSampler(label_weights_train, len(dataset_train))


        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=num_w, pin_memory=True, drop_last=True, sampler=sampler_my)
        loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_w, pin_memory=True, drop_last=False)
    else:
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_w, pin_memory=True, drop_last=True)
        loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_w, pin_memory=True, drop_last=False)
    # loader_all = DataLoader(dataset_all, batch_size=batch_size, shuffle=True, num_workers=num_w, pin_memory=True, drop_last=True)


    # model = seq_conv(5).to(device)
    model = resnet50().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learn_rate, weight_decay=1e-1)
    lr_sh = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50])

    # loss_fun = nn.CrossEntropyLoss()
    loss_fun = nn.MSELoss()
    # loss_fun = nn.SmoothL1Loss()
    # loss_fun = nn.BCELoss()
    max_f1 = 0.0
    best_epoch = 0
    for epoch in range(max_epoch):
        list_labels = []
        list_preds = []
        model.train()
        # for i, data in enumerate(loader_train):
        for data in tqdm(loader_train, ncols=100):
            data_seq,data_seq_angen, data_lbl, data_lbl_oh = data
            data_seq,data_seq_angen, data_lbl, data_lbl_oh = data_seq.to(device),data_seq_angen.to(device), data_lbl.to(device), data_lbl_oh.to(device)

            optimizer.zero_grad()
            out = model.forward(data_seq, data_seq_angen)
            loss = torch.sqrt(loss_fun(out , data_lbl_oh))
            loss.backward()
            optimizer.step()

            # print('Epoch:{}/{}, Iter:{}/{}, loss:{:.6f}, lr:{:.6f}'.format(epoch+1, max_epoch, i+1, len(loader_train), loss.item(), optimizer.param_groups[0]['lr']), end='\r')

            list_labels += data_lbl.cpu().numpy().tolist()
            # temp_preds = torch.argmax(out.detach().cpu(), dim=1)
            temp_preds = torch.sum(out.detach(), dim=1).cpu()
            list_preds += temp_preds.numpy().tolist()
        # lr_sh.step()
        # print('')
        print('train:', epoch+1)
        temp_f1 = stats.pearsonr(list_labels, list_preds)[0]
        temp_f2 = 1-rmse(list_labels, list_preds)/2

        print('  PC:', temp_f1)
        print('RMSE:', temp_f2)


        

        list_labels = []
        list_preds = []
        
        model.eval()

        torch.save(model.state_dict(), './weights/model_aff_temp.pth')

        for data in tqdm(loader_valid, ncols=100):
            data_seq,data_seq_angen, data_lbl, data_lbl_oh = data
            data_seq,data_seq_angen, data_lbl, data_lbl_oh = data_seq.to(device),data_seq_angen.to(device), data_lbl.to(device), data_lbl_oh.to(device)

            with torch.no_grad():
                out = model.forward(data_seq, data_seq_angen)


            list_labels += data_lbl.cpu().numpy().tolist()
            # temp_preds = torch.argmax(out.detach().cpu(), dim=1)
            temp_preds = torch.sum(out.detach(), dim=1).cpu()
            list_preds += temp_preds.numpy().tolist()
        

        print('valid:', epoch+1)
        
        temp_f1 = stats.pearsonr(list_labels, list_preds)[0]
        temp_f2 = 1-rmse(list_labels, list_preds)/2

        print('valid PC:', temp_f1)
        print('RMSE:', temp_f2)
        temp_com = (temp_f1 + temp_f2)/2
        if temp_com >= max_f1 and epoch+1>30:
            max_f1 = temp_com
            best_epoch = epoch
            best_pcc = temp_f1
            best_rmse = temp_f2
            print('===============================================================================best f1:', max_f1)
            torch.save(model.state_dict(), './weights/model_aff_best.pth')


    print('finish, best valid epoch:', best_epoch)
    print('best comb:',max_f1)
    print('best PCC :',best_pcc)
    print('best RMSE:',best_rmse)


            


if __name__ == '__main__':
    main_train()