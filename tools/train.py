import os
from torch.utils.data import DataLoader
from data import data_process
from trainer import common_trainer
from model.Model import R3D_18

def main():
    # 制作数据集
    root_path = os.path.abspath("../")
    batch_size = 8

    # 制作模型
    train_model = R3D_18(root_path, pretrained=False)  # 使用训练好的参数作为初始化参数

    train_dataset = data_process.VideoDataset(root_path, video_size=8, phase="train", transform=None)
    Dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    '''
        超参数集合：
        layers_need_to_train    ==  0   --> 训练所有层
                                ==  1   --> 仅训练全连接层
                                ==  2   --> 待更新...
    '''
    trainer_hyperparam = {
        'lr': 1.3e-3,
        'layers_need_to_train': 0,
        'epoches': 10,
    }

    # 开始训练
    common_trainer.train(train_model, Dataloader, **trainer_hyperparam)


if __name__ == "__main__":
    main()

