# -*- coding = utf-8 -*-
# @Time : 2023/12/5 16:25
# @Author : yxz
# @File : split.py
# @Software : PyCharm
import os
import random
import shutil


def image_sort(path_list):
    path_list = sorted(path_list, key=lambda x: (x.split(".")[0], int(x.split(".")[1].split("_")[-1][1:])))
    return path_list


def split(init_data, train_ratio, module_dir):
    train_data = []
    test_data = []
    length = len(init_data)

    for i in range(0, length, 12):
        judge = random.random()
        if judge < train_ratio:
            train_data.extend(init_data[i:i + 12])
        else:
            test_data.extend(init_data[i:i + 12])

    train_dir = module_dir + '/train'
    test_dir = module_dir + '/test'

    if not os.path.exists(module_dir):
        os.mkdir(module_dir)

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
        for data in train_data:
            shutil.copy(data, train_dir)

    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
        for data in test_data:
            shutil.copy(data, test_dir)

def main():
    add_noise = False

    class_names = ['desk', 'door', 'dresser', 'flower_pot']
    clean_path = "D:/data/40/40/clean/"  # content: bg_noise, clean, label_noise, sub_content: desk, door, dresser, flower_pot
    noise_path = "D:/data/40/40/label_noise/"
    res_path = "label_learning_images_new_12x/"

    for module in class_names:
        filepath = clean_path + module
        images = image_sort(os.listdir(filepath))
        images = [os.path.join(filepath, images[i]) for i in range(len(images))]

        if add_noise:
            filepath = noise_path + module
            images_ = image_sort(os.listdir(filepath))
            images_ = [os.path.join(filepath, images_[i]) for i in range(len(images_))]
            images = images.extend(images_)

        split(images, 0.8, res_path + module)
    print()


if __name__ == "__main__":
    main()

