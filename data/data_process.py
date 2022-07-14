import torch
import os
import json
import torchvision
import random
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class VideoDataset(Dataset):
    def __init__(self, root_path, video_size, phase="train", transform=None, **kwargs):
        super(VideoDataset, self).__init__()
        self.video_size = video_size
        self.phase = phase
        self.root_path = root_path
        self.transform = transform
        self.dataset_path = os.path.join(root_path,phase)
        self.labels = {}
        self.video_names = []
        self.targets = []
        # 检查地址是否正确
        assert os.path.exists(self.dataset_path), "Data path {:s} is not exist!".format(self.dataset_path)
        # 检查目录命名形式是否正确
        assert self.checkDir(), "Data dir is incorrect!"

        self.labels = self.getLabel()
        self.video_names , self.targets = self.getVediosAndLabels(self.dataset_path)

        # 检查视频数量是否等于标签数量
        assert len(self.targets) == len(self.video_names) , "len(Videos) != len(targets) "
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

        target = self.targets[index]
        return output_frames,target

    def __len__(self):
        return len(self.video_names)

    def getVediosAndLabels(self,data_path):
        targets = []
        video_names = []
        for label in os.listdir(data_path):
            for video_name in os.listdir(os.path.join(data_path,label)):
                video_names.append(os.path.join(data_path,label,video_name))
                targets.append(int(label))
        return video_names,targets


    def basicalTransform(self,frames):
        frames = frames.permute(0, 3, 1, 2).to(torch.float64)   # convert to 'TCHW'
        frame_list = []
        for frame in frames:
            frame = self.basic_transforms(frame)
            frame_list.append(frame)
        frames = torch.stack(frame_list,0)
        frames = frames.permute(1,0,2,3)      #convert to 'CTHW'
        return frames
    
    def getLabel(self) -> dict:
        with open(os.path.join(self.root_path, "../dataset/violent_classification.json"), 'r') as f:
            classes = f.read()
            labels = json.loads(classes)
        return labels

    def checkDir(self) -> bool:
        dir_list = os.listdir(self.data_path)
        for dir in dir_list:
            if not dir.isdigit():
                return False
        return True

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
            frame = frame.transpose(1,2,0)
            frame = frame*STD + MEAN
            frame = torch.from_numpy(np.ceil(frame)).to(torch.uint8)    # 这里打印图片之前必须要将图片转换为uint8，因为打印图片的函数当接收到float类型的图片时会自动给图片乘上225
            img_batch[frame_num] = frame
        writer[0].add_images("Show {:s} Frames:[{:s}]".format(show_what, str(batch)), torch.from_numpy(img_batch).to(torch.uint8), dataformats='NHWC')


MEAN = [0.43216, 0.394666, 0.37645]
STD = [0.22803, 0.22145, 0.216989]
