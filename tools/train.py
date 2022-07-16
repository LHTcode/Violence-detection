import os

from torch.utils.data import DataLoader
from data import data_process
from trainer import common_trainer
from model.R3D import R3D_18
from torch.utils.tensorboard import SummaryWriter
import time

def main():
    # 制作数据集
    root_path = os.path.abspath("../")

    # 制作模型
    train_model = R3D_18(root_path, pretrained=True)  # 使用训练好的参数作为初始化参数

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
        'layers_need_to_train': 0,
        'epoches': 10,
        'T_size': 8,     # 3D卷积的时间维度大小; 目前这个参数的动态调参功能还没写，处于[不可更改状态]
        'batch_size': 8,
        'eval_interval': 1  # 验证间隔，如果设置为1则代表每一次训练的epoch后面都验证一次
    }

    train_dataset = data_process.VideoDataset(root_path, T_size=trainer_hyperparam['T_size'], phase="train", transform=None, **dataset_setting)
    Dataloader = DataLoader(train_dataset, trainer_hyperparam['batch_size'], shuffle=True)
    # 开始训练
    local_time = time.localtime()
    local_time_dir = str(local_time[0]) + '.' + str(local_time[1]) \
                     + '.' + str(local_time[2]) + '.' + str(local_time[3]) \
                     + '.' + str(local_time[4]) + '.' + str(local_time[5])
    writer = SummaryWriter(log_dir=os.path.join(root_path, "runs_log", local_time_dir))

    common_trainer.train(train_model, Dataloader, writer, **trainer_hyperparam)


if __name__ == "__main__":
    main()

