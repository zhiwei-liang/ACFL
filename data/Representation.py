from torch.utils.data import Dataset

class Representation(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # x = torch.from_numpy(self.data[idx])  # 将Numpy数组转换为PyTorch张量
        x = self.data[idx]
        y = self.labels[idx]  # 获取对应的y标签
        return x, y