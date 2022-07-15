import torch
import os
import json
import torchvision
import random
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset

def showData(frames: torch.tensor, writer: list, preds=None, target=None):
    show_frames = frames.permute(0, 2, 1, 3, 4)  #convert to 'NTCWH'
    show_what = ''
    if preds is not None and target is not None:
        show_what = "Correct"
        show_frames = show_frames[preds == target]       # 这里利用了tensor的一些特性
    for batch in range(show_frames.size(dim=0)):                 # 每个batch画一张图
        img_batch = np.zeros((show_frames.size(dim=1), 112, 112, 3))    # 'THWC'
        for frame_num in range(show_frames.size(dim=1)):
            frame = show_frames[batch][frame_num].numpy()        # 取出每一帧的数据
            frame = frame.transpose(1, 2, 0)
            frame = frame*STD + MEAN
            frame = torch.from_numpy(np.ceil(frame)).to(torch.uint8)    # 这里打印图片之前必须要将图片转换为uint8，因为打印图片的函数当接收到float类型的图片时会自动给图片乘上225
            img_batch[frame_num] = frame
        writer[0].add_images("Show {:s} Frames:[{:s}]".format(show_what, str(batch)), torch.from_numpy(img_batch).to(torch.uint8), dataformats='NHWC')


def getLabel(dataset_root_path, label_file_name) -> dict:
    label_path = os.path.join(dataset_root_path, label_file_name)
    assert os.path.exists(label_path), "ERROR: Label file no found."
    with open(label_path, 'r') as f:
        classes = f.read()
        labels = json.loads(classes)
    return labels

class VideoDataset(Dataset):
    def __init__(self, root_path, T_size, phase="train", transform=None, **kwargs):
        super(VideoDataset, self).__init__()
        self.video_size = T_size
        self.phase = phase
        self.dataset_root_path = os.path.join(root_path, "dataset")
        self.label_file_name = kwargs["label_file_name"]
        if not os.path.exists(self.dataset_root_path):  # 如果没有这个目录就创建
            os.mkdir(self.dataset_root_path)
        self.transform = transform
        self.dataset_path = os.path.join(self.dataset_root_path, kwargs["dataset_name"], phase)     # RWF-2000是这个数据集的名字
        assert os.path.exists(self.dataset_path), "Dataset path {:s} is not exist!".format(self.dataset_path)
        self.labels = {}
        self.video_names = []
        self.targets = []
        # 检查地址是否正确
        # 检查目录命名形式是否正确
        assert self.checkDir(), "ERROR: Dataset dir format must be digital."

        self.labels = getLabel(self.dataset_root_path, self.label_file_name)
        self.video_names, self.targets = self.getVediosAndTargets(self.dataset_path)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # 检查视频数量是否等于标签数量
        assert len(self.targets) == len(self.video_names), "len(Videos) != len(targets) "
        self.basic_transforms = T.Compose(
            [
                T.Resize(size = [128, 171]),
                T.CenterCrop(112),
                T.Normalize(MEAN, STD)
            ]
        )

    def __getitem__(self, index):
        video = self.video_names[index]
        frames, _, _ = torchvision.io.read_video(video, pts_unit='sec')     # vedio is 'THWC'
        while (frames.size(dim=0) < self.video_size):     # 帧数小于期望值的视频就丢弃
            index = random.randint(0,self.__len__()-1)
            video = self.video_names[index]
            frames, _, _ = torchvision.io.read_video(video, pts_unit='sec')

        # 每个数据都要求取出相同的帧数，故采用间隔取样的方法
        if frames.size(dim=0) % self.video_size != 0:   # 如果帧数不能整除则去掉末尾的一些帧
            frames = frames[0: (frames.size(dim=0) // self.video_size) * self.video_size, :, :, :]
        output_frames = frames[::int(frames.size(dim=0) / self.video_size), :, :, :]        # 间隔采样的采样帧数是 处理后的总帧数/期望获得的帧数
        # 执行一些必要的 transform
        output_frames = self.basicalTransform(output_frames)

        # 执行外加的transform（这里的transform必须是针对图片的）
        if self.transform is not None:
            output_frames = self.transform(output_frames)

        target = self.targets[index].to(self.device)
        return output_frames, target

    def __len__(self):
        return len(self.video_names)

    def getVediosAndTargets(self, data_path):
        targets = []
        video_names = []
        for target in os.listdir(data_path):
            for video_name in os.listdir(os.path.join(data_path, target)):
                video_names.append(os.path.join(data_path, target, video_name))
                targets.append(torch.tensor(int(target)))
        return video_names, targets

    def basicalTransform(self,frames):
        frames = frames.permute(0, 3, 1, 2).to(torch.float64)   # convert to 'TCHW', dtype=torch.float64
        frame_list = []
        for frame in frames:
            frame = self.basic_transforms(frame)
            frame_list.append(frame)
        frames = torch.stack(frame_list,0)
        frames = frames.permute(1,0,2,3)      # convert to 'CTHW'
        return frames

    def checkDir(self) -> bool:
        dir_list = os.listdir(self.dataset_path)
        for dir in dir_list:
            if not dir.isdigit():
                return False
        return True

MEAN = [0.43216, 0.394666, 0.37645]
STD = [0.22803, 0.22145, 0.216989]