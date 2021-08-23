from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import codecs as cs


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

    ntu_action_labels = [6, 7, 8, 9, 22, 23, 24, 38, 80, 93, 99, 100, 102]

    kinect_vibe_extract_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 24, 38]
    file_prefix = "../../../action_to_motion/dataset" if data_dir is None else data_dir
    motion_desc_file = "ntu_vibe_list.txt"
    joints_num = 18
    input_size = 54
    labels = ntu_action_labels
    dataset = ImageDataset(file_prefix, motion_desc_file, labels, joints_num=joints_num,
                                            offset=True, extract_joints=None)

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
    def __init__(self, file_prefix, candi_list_desc, labels, joints_num=18, do_offset=True, extract_joints=None):
        self.data = []
        self.labels = labels
        self.lengths = []
        self.label_enc = dict(zip(labels, np.arange(len(labels))))
        self.label_enc_rev = dict(zip(np.arange(len(labels)), labels))
        candi_list = []

        candi_list_desc_name = os.path.join(file_prefix, candi_list_desc)
        with cs.open(candi_list_desc_name, 'r', 'utf-8') as f:
            for line in f.readlines():
                candi_list.append(line.strip())

        for path in candi_list:
            data_org = joblib.load(os.path.join(file_prefix, path))
            # (motion_length, 49, 3)
            # print(os.path.join(file_prefix, path))
            try:
                data_mat = data_org[1]['joints3d']
            except Exception:
                continue
            action_id = int(path[path.index('A') + 1:-4])
            motion_mat = data_mat

            if extract_joints is not None:
                # (motion_length, len(extract_joints, 3))
                motion_mat = motion_mat[:, extract_joints, :]


            # change the root keypoint of skeleton, exchange the location of 0 and 8
            #if opt.use_lie:
            # tmp = np.array(motion_mat[:, 0, :])
            # motion_mat[:, 0, :] = motion_mat[:, 8, :]
            # motion_mat[:, 8, :] = tmp

            # Locate the root joint of initial pose at origin
            if do_offset:
                offset_mat = motion_mat[0][0]
                motion_mat = motion_mat - offset_mat

            arr = cv2.resize(motion_mat, (64, 64), interpolation=cv2.INTER_CUBIC)

            self.data.append((arr, action_id))

    def get_label_reverse(self, en_label):
        return self.label_enc_rev[en_label]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        arr, label = self.data[item]
        en_label = self.label_enc[label]
        out_dict = {}
        out_dict["y"] = np.array(en_label, dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict

'''def normalize(image):
    def mul(a, b):
        return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])
    nor_img = None
    for i in range(image.shape[0]):
        # frame is 25 * 3
        frame = image[i]
        frame -= frame[1]
        x = frame[4] - frame[8]
        spine = frame[20] - frame[0]
        y = mul(x, mul(x, spine))
        if np.sum(y * spine) < 0:
            y = -y
        z = mul(x, y)
        eps = 1e-9
        # print(frame, x, y, z)
        x = x / np.sqrt(np.sum(x ** 2)) if np.sqrt(np.sum(x ** 2)) > 1e-9 else np.array([1,0,0])
        y = y / np.sqrt(np.sum(y ** 2)) if np.sqrt(np.sum(y ** 2)) > 1e-9 else np.array([0,1,0])
        z = z / np.sqrt(np.sum(z ** 2)) if np.sqrt(np.sum(z ** 2)) > 1e-9 else np.array([0,0,1])
        base = np.array([x, y, z])
        for j in range(frame.shape[0]):
            frame[j] = np.dot(frame[j], np.linalg.inv(base))
        # spine = frame[20] - frame[0]
        # div = np.sqrt(np.sum(spine ** 2))
        # frame = frame / div if div > 1e-9 else frame / (div + eps)
        # print(frame[4] - frame[8])
        if nor_img is None:
            nor_img = np.array([frame])
        else:
            nor_img = np.concatenate((nor_img, np.array([frame])), axis=0)
    return nor_img'''


'''class ImageDataset(Dataset):
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
        while i < len(self.class_len):
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
        return np.transpose(arr, [2, 0, 1]), out_dict'''
