from mvcnn_elr.tools.utils import LogWriter
from mvcnn.models.MVCNN import MVCNN, SVCNN
from mvcnn_elr.tools.ImgDataset import ImgDataset
from mvcnn.tools.Trainer import ModelNetTrainer
import time
import logging
import argparse
import torch.nn as nn
import torch.optim as optim
import torch
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(filename)s %(levelname)s %(message)s",  encoding='utf-8')

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="Name of the experiment", default="MVCNN_")
parser.add_argument("--batchSize", type=int, help="Batch size for the second stage",
                    default=8)  # it will be *12 images in each batch for mvcnn
parser.add_argument("--num_models", type=int, help="number of models per class", default=1000)
parser.add_argument("--lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("--weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("--no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("--cnn_name", type=str, help="cnn model name", default="vgg11")
parser.add_argument("--num_views", type=int, help="number of views", default=12)
parser.add_argument("--train_path", type=str, default="data/modelnet40_images_new_12x/*/train")
parser.add_argument("--val_path", type=str, default="data/modelnet40_images_new_12x/*/test")
parser.add_argument("--img_log_dir", default="mvcnn/output/logs", type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default="Model Performance ", type=str, help="Logging image file name.")
parser.add_argument("--eval_step", default=200, type=int, help="The frequency of evaluate.")
parser.add_argument("--logging_step", default=10, type=int, help="The frequency of logging print.")
parser.add_argument('--device', default="cuda:0", type=str, help="Indices of GPUs to enable (default: all).")
parser.add_argument("--save_dir", default="mvcnn/output/model", type=str, help="The path of ouput save.")
parser.add_argument("--epochs", default=30, type=int, help="The number of training epoch.")
parser.add_argument("--noise_ratio", default=0.0, type=float, help="The ratio of noise data in training.")
parser.add_argument("--image_load_path_train", type=str, default=None,
                    help="The path of train file transformed image to load.")
parser.add_argument("--image_load_path_test", type=str, default=None,
                    help="The path of test file transformed image to load.")
parser.add_argument("--image_save_path_train", type=str, default="data/image/train",
                    help="The path of train file transformed image to save.")
parser.add_argument("--image_save_path_test", type=str, default="data/image/test",
                    help="The path of test file transformed image to save.")
parser.add_argument("--image_load_path_test_multi", type=str, default=None,
                    help="The path of multi view test file transformed image to load.")
parser.set_defaults(train=False)


if __name__ == '__main__':
    args = parser.parse_args()
    pretraining = not args.no_pretraining
    classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                  'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                  'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                  'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                  'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

    # STAGE 1
    save_dir = args.save_dir + "/stage_1"
    time_stage1_start = time.time()
    writer = LogWriter(log_path=args.img_log_dir, log_name=args.img_log_name +
                       args.name + str(args.noise_ratio) + "_stage_1")
    cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name)

    optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_models_train = args.num_models*args.num_views

    train_dataset = ImgDataset(args.train_path, classnames=classnames, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=1, add_noise=True,
                               noise_ratio=args.noise_ratio, image_load_path=args.image_load_path_train, image_save_path=args.image_save_path_train)
    val_dataset = ImgDataset(args.val_path, classnames=classnames, scale_aug=False, rot_aug=False, test_mode=True, num_models=0,
                             num_views=1, add_noise=False, image_load_path=args.image_load_path_test, image_save_path=args.image_save_path_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    logging.info('num_train_files: '+str(len(train_dataset.filepaths)))
    logging.info('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(),
                              'svcnn', save_dir, num_views=1, eval_step=args.eval_step, logging_step=args.logging_step, writer=writer, device=args.device)
    trainer.train(args.epochs)

    # STAGE 2
    writer = LogWriter(log_path=args.img_log_dir, log_name=args.img_log_name +
                       args.name + str(args.noise_ratio) + "_stage_2")
    save_dir = args.save_dir + "/stage_2"
    cnet = torch.load(os.path.join(args.save_dir + "/stage_1/model_best", "model.pt"))
    cnet_2 = MVCNN(args.name, cnet, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views)
    del cnet

    optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    train_dataset = ImgDataset(args.train_path, classnames=classnames, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views, add_noise=True,
                               noise_ratio=args.noise_ratio, image_load_path=args.image_load_path_train, image_save_path=args.image_save_path_train)
    val_dataset = ImgDataset(args.val_path, classnames=classnames, scale_aug=False, rot_aug=False, test_mode=True, num_models=0,
                             num_views=args.num_views, add_noise=False, image_load_path=args.image_load_path_test_multi, image_save_path=args.image_save_path_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
    logging.info('num_train_files: '+str(len(train_dataset.filepaths)))
    logging.info('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(),
                              'mvcnn', save_dir, num_views=args.num_views, eval_step=args.eval_step, logging_step=args.logging_step, writer=writer, device=args.device)
    trainer.train(args.epochs)


# python -m mvcnn.train.train_mvcnn
