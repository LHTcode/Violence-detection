import os
import sys
from torch.utils.data import DataLoader
from data import data_process
from trainer import common_trainer
from model.R3D import R3D_18
from torch.utils.tensorboard import SummaryWriter
import time

# base = os.path.abspath("../")
# sys.path.append(base)

def main():
    # 制作数据集
    root_path = os.path.abspath("../")

    # 制作模型
    model = {x: R3D_18(root_path, pretrained=y, is_train_phase=y)
             for x, y in zip(["train", "valid"], [True, False])}
    '''
        dataset有关的设置
    '''
    dataset_setting = {
        "label_file_name": "violent_classification.json",
        "dataset_name": "RWF-2000"
    }
    '''
        超参数：
        layers_need_to_train    ==  0   --> 训练所有层
                                ==  1   --> 仅训练全连接层
                                ==  2   --> 待更新...
    '''
    trainer_hyperparam = {
        'lr': 1.3e-3,
        'step_size':10,
        'gamma':0.1,
        'layers_need_to_train': 0,
        'epoches': 30,
        'T_size': 8,     # 3D卷积的时间维度大小; 目前这个参数的动态调参功能还没写，处于[不可更改状态]
        'batch_size': 8,
        'eval_interval': 3  # 验证间隔，如果设置为1则代表每一次训练的epoch后面都验证一次
    }
    valid_hyperparam = {
        'T_size': 8,
        'batch_size': 15,
    }

    dataset = {x: data_process.VideoDataset(root_path, T_size=y, phase=x, transform=None, **dataset_setting)
                     for x, y in zip(['train', 'valid'], [trainer_hyperparam["T_size"], valid_hyperparam["T_size"]])}
    Dataloader = {x: DataLoader(dataset[x], y, shuffle=True)
                  for x ,y in zip(["train", "valid"], [trainer_hyperparam['batch_size'], valid_hyperparam['batch_size']])}
    print("Dataloader, dataset", Dataloader, dataset)
    # 开始训练
    local_time = time.localtime()
    local_time_dir = str(local_time[0]) + '.' + str(local_time[1]) \
                     + '.' + str(local_time[2]) + '.' + str(local_time[3]) \
                     + '.' + str(local_time[4]) + '.' + str(local_time[5])
    writer = SummaryWriter(log_dir=os.path.join(root_path, "runs_log", local_time_dir))

    common_trainer.train(model, Dataloader, writer, **trainer_hyperparam)


if __name__ == "__main__":
    main()

