import numpy as np
import glob
import torch.utils.data
from PIL import Image
import torch
from tqdm import tqdm
from torchvision import transforms
import logging


class ImgDataset(torch.utils.data.Dataset):

    NOISE_INFO = []
    FILE_CLASS = []

    def __init__(self, root_dir, classnames, scale_aug=False, rot_aug=False, test_mode=False, num_models=0, num_views=12,
                 add_noise=False, noise_ratio=0, image_load_path=None, image_save_path=None):
        self.root_dir = root_dir
        self.classnames = classnames
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_models = num_models
        self.num_views = num_views
        self.add_noise = add_noise
        self.noise_ratio = noise_ratio
        self.image_load_path = image_load_path
        self.image_save_path = image_save_path

        if self.num_views != 1 and self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        self.get_filepaths()

    def get_filepaths(self):
        set_ = self.root_dir.split('/')[-1]
        parent_dir = self.root_dir.rsplit('/', 2)[0]
        self.filepaths_dict = {}
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*shaded*.png'))
            if self.num_models == 0:
                # Use the whole dataset
                self.filepaths_dict[self.classnames[i]] = all_files
            else:
                self.filepaths_dict[self.classnames[i]] = all_files[:min(self.num_models, len(all_files))]

        self.filepaths = [file for all_files in self.filepaths_dict.values() for file in all_files]
        self.get_file_image()

        file_class_dict = {classname: [classname] * len(self.filepaths_dict[classname])
                           for classname in self.filepaths_dict.keys()}
        self.true_class = [self.classnames.index(fileclass)
                           for all_class in file_class_dict.values() for fileclass in all_class]

        # 循环置换标签添加噪声
        # noise_info: clean=1, noise=0
        if self.add_noise:
            # 控制stage1和stage2的噪声信息相同
            if self.NOISE_INFO == []:
                self.noise_info = {classname: [1] * len(file_class_dict[classname])
                                   for classname in file_class_dict.keys()}
                file_class = self.add_train_noise(file_class_dict)
                self.file_class = [self.classnames.index(classname) for classname in file_class]
                ImgDataset.NOISE_INFO = self.noise_info
                ImgDataset.FILE_CLASS = self.file_class
            else:
                self.noise_info = ImgDataset.NOISE_INFO
                self.file_class = ImgDataset.FILE_CLASS
                logging.info("Keep consistent with STAGE1")
        else:
            self.noise_info = [1] * len(self.filepaths)
            self.file_class = self.true_class

    def get_file_image(self):

        if self.image_load_path == "":
            self.file_image = []
            for filepath in tqdm(self.filepaths, desc=f"get {self.root_dir.split('/')[-1]} files information: "):
                image = Image.open(filepath).convert('RGB')
                if self.transform:
                    self.file_image.append(self.transform(image))

            if self.image_save_path != "":
                logging.info(f"data saving..., path: {self.image_save_path}")
                save_array = [image.numpy() for image in self.file_image]
                np.savez_compressed(self.image_save_path, *save_array)
                logging.info(f"data saved complate, path: {self.image_save_path}")
        else:
            logging.info(f"data loading..., path: {self.image_load_path}")
            self.file_image = [torch.from_numpy(image) for image in np.load(self.image_load_path).values()]
            logging.info(f"data loaded complate, path: {self.image_load_path}")

    def add_train_noise(self, file_class_dict):
        # 训练数据信息
        class_names = list(file_class_dict.keys())
        class_numbs = [int(len(all_files) / self.num_views) for all_files in file_class_dict.values()]

        # 计算需要置换标签的样本量
        total_files_num = sum(class_numbs)
        noise_files_num = round(total_files_num * self.noise_ratio)

        # 随机生成类别噪声index
        noise_index = []
        noise_numbs = []
        for class_numb in class_numbs:
            noise_i_num = round(noise_files_num * class_numb / total_files_num)
            index = list(np.random.choice(np.arange(0, (class_numb - 1) *
                                                    self.num_views, self.num_views), size=noise_i_num, replace=False))
            noise_index.append(index)
            noise_numbs.append(noise_i_num)

        # 类别循环
        for idx in range(len(class_names)):
            # 当前类别需要置换的样本数
            label_i = class_names[idx]
            noise_i_num = len(noise_index[idx])

            position = 0
            noise_j_num = 0
            for other_class in range(0, len(class_names)):
                if other_class == idx:
                    continue
                # idx需要置换的样本数中来自other_class的数量
                label_j = class_names[other_class]
                noise_j_num += round(noise_i_num * class_numbs[other_class] / (total_files_num - class_numbs[idx]))

                for _ in range(position, noise_j_num):
                    if noise_index[idx] != [] and noise_index[other_class] != []:
                        noise_index_i = noise_index[idx].pop()
                        noise_index_j = noise_index[other_class].pop()
                    else:
                        break

                    file_class_dict[label_i][noise_index_i: noise_index_i + self.num_views], file_class_dict[label_j][noise_index_j: noise_index_j + self.num_views] \
                        = file_class_dict[label_j][noise_index_j: noise_index_j + self.num_views], file_class_dict[label_i][noise_index_i:noise_index_i+self.num_views]
                    self.noise_info[label_i][noise_index_i: noise_index_i + self.num_views] = [0] * self.num_views
                    self.noise_info[label_j][noise_index_j: noise_index_j + self.num_views] = [0] * self.num_views
                    position += 1

        self.noise_info = [info for all_info in self.noise_info.values() for info in all_info]
        noise_dict = {file_class: len(file_class_dict[file_class]) - file_class_dict[file_class].count(file_class)
                      for file_class in file_class_dict.keys()}
        logging.info(
            f"Every class noise ratio reslut: {[ noise_dict[file_class] / len(file_class_dict[file_class]) for file_class in file_class_dict.keys()]}")
        logging.info(f"Noise ratio: {sum(noise_dict.values()) / (total_files_num * self.num_views)}")

        return [fileclass for all_class in file_class_dict.values() for fileclass in all_class]

    def __len__(self):
        return int(len(self.filepaths)/self.num_views)

    def __getitem__(self, idx):
        if self.num_views == 1:
            return (self.file_class[idx], self.file_image[idx], self.filepaths[idx], self.noise_info[idx], self.true_class[idx])
        else:
            return (torch.tensor(self.file_class[idx*self.num_views:(idx+1)*self.num_views]), torch.stack(self.file_image[idx*self.num_views:(idx+1)*self.num_views]),
                    self.filepaths[idx*self.num_views:(idx+1)*self.num_views], torch.tensor(self.noise_info[idx*self.num_views:(idx+1)*self.num_views]), torch.tensor(self.true_class[idx*self.num_views:(idx+1)*self.num_views]))
