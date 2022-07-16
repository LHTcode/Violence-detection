import os
import sys
from torch.utils.data import DataLoader

base = os.path.abspath("../")
sys.path.append(base)
from data import data_process
from trainer import common_trainer
from model.R3D import R3D_18
from torch.utils.tensorboard import SummaryWriter
import time
import torch


#接收模型和数据，进行测试
def valid_model (model,Dataloader):
    print("Validating...")
    #模型的路径
    root_path = os.path.abspath("../")
    model_path = root_path+"./models_parameters/2022.7.16/best_model.pt"
    model.eval()
    device = model.device
    model_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(model_dict["state_dict"])
    print("模型加载完成")

    #进行验证
    data_num = 0
    correct_num = 0
    for data in Dataloader:
        #视频对应的原标签
        video, target = data
        print("target",target)
        data_num += video.size(dim=0)
        target = torch.tensor(target).to(torch.long).to(device)

        with torch.set_grad_enabled(False):
            # 计算输出
            output = model(video)
        _, preds = torch.max(output, dim=1)
        print(output)
        print("预测",preds)
        correct_num += torch.sum(preds == target)
    precision = correct_num / data_num * 100  # 计算一个epoch下来的精度
    print("Precision = {}%".format(precision))





if __name__ == "__main__":
    #数据预处理
    dataset_setting = {
        "label_file_name": "violent_classification.json",
        "dataset_name": "RWF-2000"}
    root_path = os.path.abspath("../")
    valider_hyperparam = {
        # 'lr': 1.3e-3,
        # 'layers_need_to_train': 0,
        'epoches': 10,
        'T_size': 8,  # 3D卷积的时间维度大小; 目前这个参数的动态调参功能还没写，处于[不可更改状态]
        'batch_size': 8,
        'eval_interval': 1  # 验证间隔，如果设置为1则代表每一次训练的epoch后面都验证一次
    }
    valid_dataset = data_process.VideoDataset(root_path, T_size=valider_hyperparam['T_size'], phase="train",
                                              transform=None, **dataset_setting)
    Dataloader = DataLoader(valid_dataset, valider_hyperparam['batch_size'], shuffle=True)
    print("迭代",len(Dataloader))

    root_path = os.path.abspath("../")
    print(root_path)
    model =  R3D_18(root_path, is_train_phase=False, pretrained=False)
    #调用函数进行验证
    valid_model(model,Dataloader)
