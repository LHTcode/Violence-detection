import os
import sys

base = os.path.abspath("../")
sys.path.append(base)

from torch.utils.data import DataLoader
from data import data_process
from trainer import common_trainer
from model.R3D import R3D_18
from torch.utils.tensorboard import SummaryWriter
import time
import torch
from tqdm import tqdm


#接收模型和数据，进行测试
def valid_model (model,Dataloader,writer,**kwargs):
    print("Validating...")
    #模型的路径
    root_path = os.path.abspath("../")
    model_dir = "models_parameters"
    model_subfolder = "2022.7.16"
    model_name = "best_model.pt"
    model_path = os.path.join(root_path, model_dir,model_subfolder,model_name)
    assert os.path.exists(model_path), "model path {:s} may be wrong!".format(model_path)
    model.eval()
    device = model.device
    model_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(model_dict["state_dict"])

    data_loop = tqdm(Dataloader, total=len(Dataloader))
    epoch_loop = tqdm(range(kwargs['epoches']), total=kwargs['epoches'])
    #进行验证
    data_num = 0
    correct_num = 0
    for epoch in epoch_loop:
        print("Epoch:{:}".format(epoch))
        for data in data_loop:
            #视频对应的原标签
            video, target = data
            # print("target",target)
            data_num += video.size(dim=0)
            target = torch.tensor(target)
            with torch.set_grad_enabled(False):
                # 计算输出
                output = model(video)
            _, preds = torch.max(output, dim=1)
            correct_num += torch.sum(preds == target)
        precision = correct_num / data_num * 100  # 计算一个epoch下来的精度
        print("Precision = {}%".format(precision))
        #记录precision
        writer.add_scalar("{:s}/Precision".format("Valid"), precision, epoch)
    writer.close()




if __name__ == "__main__":
    #数据预处理
    dataset_setting = {
        "label_file_name": "violent_classification.json",
        "dataset_name": "RWF-2000"}
    root_path = os.path.abspath("../")
    valider_hyperparam = {
        'epoches': 10,
        'T_size': 8,  # 3D卷积的时间维度大小; 目前这个参数的动态调参功能还没写，处于[不可更改状态]
        'batch_size': 8,
        'eval_interval': 1  # 验证间隔，如果设置为1则代表每一次训练的epoch后面都验证一次
    }
    valid_dataset = data_process.VideoDataset(root_path, T_size=valider_hyperparam['T_size'], phase="train",
                                              transform=None, **dataset_setting)
    Dataloader = DataLoader(valid_dataset, valider_hyperparam['batch_size'], shuffle=True)

    root_path = os.path.abspath("../")
    model =  R3D_18(root_path, is_train_phase=False, pretrained=False)
    #wirter 的初始化
    local_time = time.localtime()
    local_time_dir = str(local_time[0]) + '.' + str(local_time[1]) \
                     + '.' + str(local_time[2]) + '.' + str(local_time[3]) \
                     + '.' + str(local_time[4]) + '.' + str(local_time[5])
    writer = SummaryWriter(log_dir=os.path.join(root_path, "runs_log", local_time_dir))

    #调用函数进行验证
    valid_model(model,Dataloader,writer,**valider_hyperparam)
