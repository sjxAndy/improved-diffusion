from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import cv2


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    # all_files = os.listdir(data_dir)

    if not os.path.exists('../labels.txt'):
        from tqdm import tqdm
        npys = os.listdir(data_dir)
        labels = dict()

        for name in tqdm(npys):
            action = int(name[17:20])
            if action not in labels.keys():
                labels[action] = []
            labels[action].append(name)

        newlabel = ''
        for _ in np.arange(1, 121):
            newlabel += str(_) + '\n'
            for label in (labels[_]):
                newlabel += label + '\n' 

        f = open("./labels.txt",'w')
        f.write(newlabel)


    labels = dict()
    with open('./labels.txt', 'r') as f:
        curr_lab = 0
        for line in f.readlines():
            line = line.strip('\n')
            if 0 < len(line) < 4:
                # select one class only
                if len(labels.keys()) > 1:
                    break
                labels[int(line)] = []
                curr_lab = int(line)
                continue
            labels[curr_lab].append(line)

    classes = False
    dataset = ImageDataset(
        image_size,
        data_dir,
        labels,
        classes=classes
    )
    print('num of data:', len(dataset))
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


class ImageDataset(Dataset):
    def __init__(self, resolution, data_dir, labels, classes=False):
        super().__init__()
        self.resolution = resolution
        self.data_dir = data_dir
        self.labels = labels
        self.classes = classes
        self.class_len = [len(self.labels[k]) for k in self.labels.keys()]

    def __len__(self):
        return sum(self.class_len)

    def __getitem__(self, idx):
        assert idx < sum(self.class_len)
        sum_idx, i = 0, 0
        while i < len(self.class_len):
            if sum_idx + self.class_len[i] > idx:
                break
            sum_idx += self.class_len[i]
            i += 1
        
        path = self.labels[i + 1][idx - sum_idx]

        data = np.load(os.path.join(self.data_dir, path), allow_pickle=True).item()
        image = data['skel_body0']
        image = image.astype(np.float32)
        arr = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC)

        out_dict = {}
        if self.classes:
            out_dict["y"] = np.array(i + 1, dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict
