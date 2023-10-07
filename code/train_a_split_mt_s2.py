import pandas as pd
from train_config_s2 import *
import os
import warnings
warnings.filterwarnings('ignore')

scale = 5.0


def main_train():
    num_sample = 30 ##%
    data_dir = './user_data/data_s2/Affinity_train_data.csv'
    data_ex_dir = './user_data/data_s2/Affinity_train_data_ex.csv'

    # data_dir = 'D:/proj_SARS-CoV-2_baseline/data/train_data_full_new.csv'
    # data_ex_dir = 'D:/proj_SARS-CoV-2_baseline/data/train_data_full_ex_new.csv'


    

    data = pd.read_csv(data_dir)
    data_seq = data['Sequence'].values.tolist()
    data_lbl = data['Label'].values.tolist()


    data_ex = pd.read_csv(data_ex_dir)
    data_seq_ex = data_ex['Sequence'].values.tolist()
    data_lbl_ex = data_ex['Label'].values.tolist()


    # data_seq_train = data_seq[num_sample:]
    # data_lbl_train = data_lbl[num_sample:]
    # data_seq_valid = data_seq[:num_sample]
    # data_lbl_valid = data_lbl[:num_sample]


    # data_seq_train = data_seq
    # data_lbl_train = data_lbl
    # data_seq_valid = data_seq_ex
    # data_lbl_valid = data_lbl_ex


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

    

    # data_seq_train, data_lbl_train = resample_dataset(data_seq_train, data_lbl_train)
    data_seq_valid, data_lbl_valid = resample_dataset(data_seq_valid, data_lbl_valid)
    

    ## load pdb structure data
    data_str_dir = './user_data/data/data_str/'
    list_names = os.listdir(data_str_dir)
    data_str =[]
    for temp_name in tqdm(list_names, ncols=100):
        data_str.append(data_str_dir + temp_name)
        # temp_np = np.load(data_str_dir + temp_name, allow_pickle=True)
        # data_str.append(temp_np)




    dataset_train = dataset_anti_mt_str(data_seq_train, data_lbl_train, data_str , 'train')
    dataset_valid = dataset_anti_mt_str(data_seq_valid, data_lbl_valid, data_str ,'valid')
    print('dataset init finished')



    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_w, pin_memory=True, drop_last=True)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=num_w, pin_memory=True, drop_last=False)
    # loader_all = DataLoader(dataset_all, batch_size=batch_size, shuffle=True, num_workers=num_w, pin_memory=True, drop_last=True)

    
    max_f1 = 0.0
    best_epoch = 0
    for epoch in range(max_epoch):
        list_labels = []
        list_preds = []
        model.train()
        for i, data in enumerate(loader_train):
        # for data in tqdm(loader_train, ncols=50):
            data_seq,data_seq_angen, data_lbl, data_lbl_oh, \
                    data_ch_loc, data_cl_loc,\
                         map_seq, map_str = data
            data_seq,data_seq_angen, data_lbl, data_lbl_oh = data_seq.to(device),data_seq_angen.to(device), data_lbl.to(device), data_lbl_oh.to(device)
            data_ch_loc, data_cl_loc = data_ch_loc.to(device), data_cl_loc.to(device)
            map_seq, map_str = map_seq.to(device), map_str.to(device)


            optimizer.zero_grad()
            out = model.forward_train_mt(data_seq, data_seq_angen)
            out_p, out_ch, out_cl = out
            loss_p = loss_fun(out_p, data_lbl_oh)
            loss_ch = loss_fun_str(out_ch, data_ch_loc)
            loss_cl = loss_fun_str(out_cl, data_cl_loc)
            loss_loc = (loss_ch + loss_cl)/2.0
            loss = loss_p + loss_loc
            # loss.backward()
            # optimizer.step()
            
            # optimizer.zero_grad()
            out_h, out_a = model.forward_train_str(map_seq)
            loss_h = loss_fun_str(out_h*scale, map_str*scale)
            loss_a = loss_fun_str(out_a*scale, map_str*scale)
            loss_str = (loss_h + loss_a)/2.0
            # loss_str.backward()

            loss_all = loss + loss_str
            loss_all.backward()
            optimizer.step()
            
            


            print('Epoch:{}/{}, Iter:{}/{}, loss:{:.6f}, loss_str:{:.6f}, lr:{:.6f}'.format(epoch+1, max_epoch, i+1, len(loader_train), loss.item(), loss_str.item(),optimizer.param_groups[0]['lr']), end='\r')

            list_labels += data_lbl.cpu().numpy().tolist()
            temp_preds = torch.sum(out_p.detach(), dim=1).cpu().view(-1).numpy()
            list_preds += (temp_preds+0.5).astype(np.int8).tolist()
        lr_sh.step()
        print('')
        print('train:', epoch+1)
        train_pcc = stats.pearsonr(list_labels, list_preds)[0]
        train_rmse = 1-rmse(list_labels, list_preds)/2

        print('  PC:', train_pcc)
        print('RMSE:', train_rmse)



        

        list_labels = []
        list_preds = []
        
        model.eval()
        torch.save(model.state_dict(),  model_save_dir + '/model_aff_mt_str_temp.pth')

        model_eval.load_state_dict(torch.load(model_save_dir + '/model_aff_mt_str_temp.pth', map_location='cpu'), strict=True)
        model_eval.eval()

        for data in tqdm(loader_valid, ncols=50):
            data_seq,data_seq_angen, data_lbl, data_lbl_oh = data
            data_seq,data_seq_angen, data_lbl, data_lbl_oh = data_seq.to(device),data_seq_angen.to(device), data_lbl.to(device), data_lbl_oh.to(device)

            with torch.no_grad():
                out = model_eval.forward(data_seq, data_seq_angen)


            list_labels += data_lbl.cpu().numpy().tolist()
            temp_preds = torch.sum(out.detach(), dim=1).cpu().view(-1).numpy()
            list_preds += (temp_preds+0.5).astype(np.int8).tolist()
        

        print('valid:', epoch+1)
        temp_f1 = stats.pearsonr(list_labels, list_preds)[0]
        temp_f2 = 1-rmse(list_labels, list_preds)/2
        temp_re = metrics.classification_report(list_labels, list_preds)
        print(Counter(list_labels))
        print(Counter(list_preds))
        print('valid PC:', temp_f1)
        print('    RMSE:', temp_f2)
        print(temp_re)
        # temp_com = (temp_f1 + temp_f2)/2
        temp_com = temp_f1
        if temp_com >= max_f1 and epoch+1>20:
            max_f1 = temp_com
            best_epoch = epoch
            best_pcc = temp_f1
            best_rmse = temp_f2
            best_train = train_pcc
            print('===============================================================================best f1:', max_f1)
            torch.save(model_eval.state_dict(), model_save_dir + 'model_aff_mt_str_best.pth')


    print('finish, best valid epoch:', best_epoch)
    print('best comb:',max_f1)
    print('best PCC :',best_pcc)
    print('best RMSE:',best_rmse)
    print('train PCC:', best_train)




if __name__ == '__main__':
    main_train()