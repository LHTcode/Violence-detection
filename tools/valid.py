import os
import sys
from torch.utils.data import DataLoader
from data import data_process
from model.R3D import R3D_18
from torch.utils.tensorboard import SummaryWriter
import time
import torch
from tqdm import tqdm

# base = os.path.abspath("../")
# sys.path.append(base)

# 接收模型和数据，进行测试
def valid(model, Dataloader, writer, **kwargs):
    print("\nValidating...")
    model.eval()
    device = model.device
    model_dict = torch.load(model.getSaveDir(), map_location=torch.device(device))
    model.load_state_dict(model_dict["state_dict"])

    # 进行验证
    data_num = 0
    correct_num = 0
    for data in Dataloader:
        video, target = data
        data_num += video.size(dim=0)
        with torch.set_grad_enabled(False):
            # 计算输出
            output = model(video)
        _, preds = torch.max(output, dim=1)
        correct_num += torch.sum(preds == target)
    precision = correct_num / data_num * 100  # 计算一个epoch下来的精度
    print("\nValid Precision = {}%".format(precision))
    # 记录precision
    writer.add_scalar("Valid/Precision", precision)
    writer.close()


if __name__ == "__main__":
    #数据预处理
    dataset_setting = {
        "label_file_name": "violent_classification.json",
        "dataset_name": "RWF-2000"}
    root_path = os.path.abspath("../")
    valid_hyperparam = {
        'T_size': 8,  # 3D卷积的时间维度大小; 目前这个参数的动态调参功能还没写，处于[不可更改状态]
        'batch_size': 8,
    }
    valid_dataset = data_process.VideoDataset(root_path, T_size=valid_hyperparam['T_size'], phase="valid",
                                              transform=None, **dataset_setting)
    Dataloader = DataLoader(valid_dataset, valid_hyperparam['batch_size'], shuffle=True)

    model =  R3D_18(root_path, is_train_phase=False, pretrained=False)
    # wirter的初始化
    local_time = time.localtime()
    local_time_dir = str(local_time[0]) + '.' + str(local_time[1]) \
                     + '.' + str(local_time[2]) + '.' + str(local_time[3]) \
                     + '.' + str(local_time[4]) + '.' + str(local_time[5])
    writer = SummaryWriter(log_dir=os.path.join(root_path, "runs_log", local_time_dir))

    #调用函数进行验证
    valid(model, Dataloader, writer, **valid_hyperparam)
