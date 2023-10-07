import pandas as pd
from train_config_s3 import *
import os

def main_train():
    num_sample = 30 ##%
    data_dir = './user_data/data_s2/Neutralization_train_data.csv'

    # data_dir = 'D:/proj_SARS-CoV-2_baseline/data/train_data_full_new.csv'
    # data_ex_dir = 'D:/proj_SARS-CoV-2_baseline/data/train_data_full_ex_new.csv'


    

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



    data_seq_train, data_lbl_train = resample_dataset(data_seq_train, data_lbl_train)
    data_seq_valid, data_lbl_valid = resample_dataset(data_seq_valid, data_lbl_valid)

    ## load pdb structure data
    data_str_dir = './data/data_str/'
    list_names = os.listdir(data_str_dir)
    data_str =[]
    for temp_name in list_names:
        temp_np = np.load(data_str_dir + temp_name, allow_pickle=True)
        data_str.append(temp_np)



    dataset_train = dataset_anti_graph(data_seq_train[:], data_lbl_train, 'train')
    dataset_valid = dataset_anti_graph(data_seq_valid, data_lbl_valid, 'valid')
    print('dataset init finished')



    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_w, drop_last=True)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=num_w, drop_last=False)

    
    max_f1 = 0.0
    best_epoch = 0
    for epoch in range(max_epoch):
        list_labels = []
        list_preds = []
        model.train()
        # for i, data in enumerate(loader_train):
        for data in tqdm(loader_train, ncols=50):
            n_h, e_h, n_l, e_l, n_g, e_g, lbl, lbl_oh = data
            
            n_h, e_h = n_h.to(device), e_h.to(device)
            n_l, e_l = n_l.to(device), e_l.to(device)
            n_g, e_g = n_g.to(device), e_g.to(device)
            lbl, lbl_oh = lbl.to(device), lbl_oh.to(device)

            
            optimizer.zero_grad()
            out = model.forward(n_h, e_h, n_l, e_l, n_g, e_g)
            loss_p = loss_fun(out, lbl_oh)
            loss = loss_p
            loss.backward()
            optimizer.step()


            # print('Epoch:{}/{}, Iter:{}/{}, loss:{:.6f}, lr:{:.6f}'.format(epoch+1, max_epoch, i+1, len(loader_train), loss.item(),optimizer.param_groups[0]['lr']), end='\r')

            list_labels += lbl.cpu().numpy().tolist()
            temp_preds = torch.sum(out.detach(), dim=1).cpu().view(-1).numpy()
            list_preds += (temp_preds+0.5).astype(np.int8).tolist()
        lr_sh.step()
        # print('')
        print('train:', epoch+1)
        train_pcc = stats.pearsonr(list_labels, list_preds)[0]
        train_rmse = 1-rmse(list_labels, list_preds)/2

        print('Train PC:', train_pcc)
        print('RMSE:', train_rmse)


        

        list_labels = []
        list_preds = []
        
        model.eval()
        torch.save(model.state_dict(),  model_save_dir + '/model_neu_gcn_temp.pth')

        model_eval.load_state_dict(torch.load(model_save_dir + '/model_neu_gcn_temp.pth', map_location='cpu'), strict=True)
        model_eval.eval()

        for data in tqdm(loader_valid, ncols=50):
            n_h, e_h, n_l, e_l, n_g, e_g, lbl, lbl_oh = data
            
            n_h, e_h = n_h.to(device), e_h.to(device)
            n_l, e_l = n_l.to(device), e_l.to(device)
            n_g, e_g = n_g.to(device), e_g.to(device)
            lbl, lbl_oh = lbl.to(device), lbl_oh.to(device)
            
            with torch.no_grad():
                out = model.forward(n_h, e_h, n_l, e_l, n_g, e_g)

            list_labels += lbl.cpu().numpy().tolist()
            temp_preds = torch.sum(out.detach(), dim=1).cpu().view(-1).numpy()
            list_preds += (temp_preds+0.5).astype(np.int8).tolist()
        

        print('valid:', epoch+1)
        temp_f1 = stats.pearsonr(list_labels, list_preds)[0]
        temp_f2 = 1-rmse(list_labels, list_preds)/2
        print(Counter(list_labels))
        print(Counter(list_preds))
        print('Valid PC:', temp_f1)
        print('RMSE:', temp_f2)
        # temp_com = (temp_f1 + temp_f2)/2
        temp_com = temp_f1
        if temp_com >= max_f1 and epoch+1>5:
            max_f1 = temp_com
            best_epoch = epoch
            best_pcc = temp_f1
            best_rmse = temp_f2
            best_train = train_pcc
            print('===============================================================================best f1:', max_f1)
            torch.save(model_eval.state_dict(), model_save_dir + 'model_neu_gcn_best.pth')


    print('finish, best valid epoch:', best_epoch)
    print('best comb:',max_f1)
    print('best PCC :',best_pcc)
    print('best RMSE:',best_rmse)
    print('train PCC:', best_train)




if __name__ == '__main__':
    main_train()