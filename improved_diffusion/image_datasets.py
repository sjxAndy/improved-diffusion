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

    from tqdm import tqdm
    npys = os.listdir(data_dir)
    labels = dict()

    for name in tqdm(npys):
        action = int(name[17:20])
        if action not in labels.keys():
            labels[action] = []
        labels[action].append(name)

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

def normalize(image):
    def mul(a, b):
        return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])
    nor_img = None
    for i in range(image.shape[0]):
        frame = image[i]
        frame -= frame[1]
        x = frame[4] - frame[8]
        spine = frame[20] - frame[0]
        y = mul(x, mul(x, spine))
        if np.sum(y * spine) < 0:
            y = -y
        z = mul(x, y)
        x = x / np.sqrt(np.sum(x ** 2)) if np.sqrt(np.sum(x ** 2)) > 1e-9 else np.array([1,0,0])
        y = y / np.sqrt(np.sum(y ** 2)) if np.sqrt(np.sum(y ** 2)) > 1e-9 else np.array([0,1,0])
        z = z / np.sqrt(np.sum(z ** 2)) if np.sqrt(np.sum(z ** 2)) > 1e-9 else np.array([0,0,1])
        base = np.array([x, y, z])
        for j in range(frame.shape[0]):
            frame[j] = np.dot(frame[j], np.linalg.inv(base))
        # print(frame[4] - frame[8])
        if nor_img is None:
            nor_img = np.array([frame])
        else:
            nor_img = np.concatenate((nor_img, np.array([frame])), axis=0)
    return nor_img

class ImageDataset(Dataset):
    def __init__(self, resolution, data_dir, labels, classes=False):
        super().__init__()
        self.resolution = resolution
        self.data_dir = data_dir
        self.labels = labels
        self.classes = classes
        self.class_len = [len(self.labels[k]) for k in self.labels.keys()]
        self.class_keys = [k for k in self.labels.keys()]

    def __len__(self):
        return sum(self.class_len)

    def __getitem__(self, idx):
        assert idx < sum(self.class_len)
        sum_idx, i = 0, 0
        while sum_idx < len(self.class_len):
            if sum_idx + self.class_len[i] > idx:
                break
            sum_idx += self.class_len[i]
            i += 1
        
        path = self.labels[self.class_keys[i]][idx - sum_idx]

        data = np.load(os.path.join(self.data_dir, path), allow_pickle=True).item()
        image = data['skel_body0']
        image = image.astype(np.float32)
        image = normalize(image)
        arr = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC)

        out_dict = {}
        if self.classes:
            out_dict["y"] = np.array(i + 1, dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict
