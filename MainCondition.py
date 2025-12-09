import numpy as np
import os
from DiffusionFreeGuidence.TrainCondition import train, sample
from args import args
from eva_regressor import predictive_score_metrics
from eva_distance import cal_dist_dtw, cal_dist_fd
from eva_classifier import discrimative_score_metrics
import sys
import data.CMAPSSDataset as CMAPSSDataset
import wandb
from utils import wandb_record

os.environ["WANDB_MODE"] = "offline"

wandb.init(
    project="xxxxx",#revise by yourself
    tags=['EXP-compare'],
    config=args
    )
rmse_list,score_list,acc_list = [],[],[]

if __name__ == '__main__':
    if len(sys.argv)==1:
        print('-------no prompt--------')
        args.epoch = 50
        args.dataset = 'FD001'
        args.lr = 2e-3
        args.state = 'all' # train,sample,eval
        args.model_name = 'DiffUnet' 
        args.T = 500
        args.window_size = 48 
        args.sample_type = 'ddpm' # ddim, ddpm
        args.input_size = 14


    args.model_path = os.path.join("weights", f"{args.model_name}_{args.dataset}_{args.window_size}.pth")

    # 合成数据保存路径
    args.syndata_path = os.path.join("weights", "syn_data", f"syn_{args.dataset}_{args.model_name}_{args.window_size}_{args.sample_type}.npz")

    # 确保父目录存在
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.syndata_path), exist_ok=True)
    train_loop = 1

    datasets = CMAPSSDataset.CMAPSSDataset(fd_number=args.dataset, sequence_length=args.window_size ,deleted_engine=[1000])
    train_data = datasets.get_train_data()
    
    train_data,train_label = datasets.get_sensor_slice(train_data), datasets.get_label_slice(train_data)    
    test_data = datasets.get_test_data()
    test_data,test_label = datasets.get_last_data_slice(test_data) 
    
    if args.state == "train" or args.state == "all":
        train(args,train_data,train_label)
        #sample(args,train_label)
    elif args.state == "sample" or args.state == "all":
        sample(args,train_label)
    if args.state == "eval" or args.state == "all":

        loaded_data = np.load(args.syndata_path)
        pred_data = loaded_data['data']
        original_data_test = {'data':test_data,'label':test_label}
        original_data_train = {'data':train_data,'label':train_label}
        
        if args.dataset.endswith("1") or args.dataset.endswith("3"):
            #random_indices = np.random.choice(train_data.shape[0], size=len(train_data) // 10 , replace=False)
            random_indices = np.random.choice(5000, size=5000 , replace=False)
        else:
            #random_indices = np.random.choice(train_data.shape[0], size=len(train_data) // 10 , replace=False)
            random_indices = np.random.choice(5000, size=5000 , replace=False)
        print(random_indices.shape)
        

        for i in range(train_loop):
            rmse,score,acc = 0,0,0
            rmse,score= predictive_score_metrics(args, original_data_test, loaded_data)
            rmse_list.append(rmse);score_list.append(score);acc_list.append(acc)

            
        dtw_dist = cal_dist_dtw(train_data[random_indices].numpy(), pred_data[random_indices])
        fd_dist = cal_dist_fd(train_data[random_indices].numpy(), pred_data[random_indices])
        
        print("loss_list",rmse_list,"score list",score_list,"acc list",acc_list)
        print("dtw_dist",dtw_dist,"fd_dist",fd_dist)
        wandb_record(rmse_list, score_list, acc_list)
        wandb.run.summary["dtw_dist"] = dtw_dist
        wandb.run.summary["fd_dist"] = fd_dist
