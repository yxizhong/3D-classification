from mvcnn_elr.models.GVCNN import GVCNN
from mvcnn_elr.models.MVCNN import MVCNN, SVCNN
from mvcnn_elr.tools.utils import LogWriter
from mvcnn_elr.tools.ImgDataset import ImgDataset
from mvcnn_elr.tools.Trainer import ModelNetTrainer
import mvcnn_elr.ELR.model.loss as module_loss
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import time
import argparse
import os
import sys
sys.path.insert(0, "/workspace/yangxizhong/mvcnn")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(filename)s %(levelname)s %(message)s",  encoding='utf-8')

parser = argparse.ArgumentParser(description='PyTorch Template.')
parser.add_argument("--name", type=str, default="MVCNN_ELR_", help="Name of the experiment.")
parser.add_argument("--batchSize", type=int, default=8, help="Batch size for the second stage.")
parser.add_argument("--num_models", type=int, default=1000, help="Number of models per class.")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
parser.add_argument("--no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("--cnn_name", type=str, default="vgg11", help="Cnn model name.")
parser.add_argument("--num_views", type=int, default=12, help="Number of views.")
parser.add_argument("--img_log_dir", default="mvcnn_elr/output/gvcnn/logs", type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default="Model Performance ", type=str, help="Logging image file name.")
parser.add_argument("--monitor", default="max val_my_metric", type=str, help="The monitor of model effect.")
parser.add_argument("--beta", default=0.7, type=float, help="The weight beta of elr.")
parser.add_argument("--lambda_", default=3.0, type=float, help="The weight lambda of elr.")
parser.add_argument('--device', default="cuda:0", type=str, help="Indices of GPUs to enable (default: all).")
parser.add_argument("--save_dir", default="mvcnn_elr/output/mvcnn/model_20", type=str, help="The path of ouput save.")
parser.add_argument("--epochs", default=30, type=int, help="The number of training epoch.")
parser.add_argument("--eval_step", default=200, type=int, help="The frequency of evaluate.")
parser.add_argument("--logging_step", default=10, type=int, help="The frequency of logging print.")
parser.add_argument("--early_stop", default=np.inf, type=int, help="Control the training epoch early stop.")
parser.add_argument("--train_path", type=str, default="data/modelnet40_images_new_12x/*/train",
                    help="The path of train data.")
parser.add_argument("--val_path", type=str, default="data/modelnet40_images_new_12x/*/test",
                    help="The path of eval data.")
parser.add_argument("--noise_path", type=str, default=None, help="The path of noise data added to training.")
parser.add_argument("--noise_ratio", type=float, default=0.2, help="The ratio of noise data in training.")
parser.add_argument("--image_load_path_train", type=str, default="data/image/train_image.npz",
                    help="The path of train file transformed image to load.")
parser.add_argument("--image_load_path_test", type=str, default="data/image/test_image.npz",
                    help="The path of test file transformed image to load.")
parser.add_argument("--image_save_path_train", type=str, default="data/image/train_image.npz",
                    help="The path of train file transformed image to save.")
parser.add_argument("--image_save_path_test", type=str, default="data/image/test_image.npz",
                    help="The path of test file transformed image to save.")
parser.add_argument("--image_load_path_test_multi", type=str, default="data/image/test_image_enhanced.npz",
                    help="The path of multi view test file transformed image to load.")
parser.set_defaults(train=False)
args = parser.parse_args()


def train_stage1():
    time_stage1_start = time.time()
    writer = LogWriter(log_path=args.img_log_dir, log_name=args.img_log_name +
                       args.name + str(args.noise_ratio) + "_stage_1")
    # STAGE 1
    save_dir = args.save_dir + '/stage_1'
    cnet = SVCNN(args.name, nclasses=n_classes, pretraining=pretraining,
                 cnn_name=args.cnn_name, classnames=classnames)
    optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_dataset = ImgDataset(args.train_path, classnames=classnames, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=1, add_noise=True,
                               noise_ratio=args.noise_ratio, image_load_path=args.image_load_path_train, image_save_path=args.image_save_path_train)
    val_dataset = ImgDataset(args.val_path, classnames=classnames, scale_aug=False, rot_aug=False, test_mode=True, num_models=0,
                             num_views=1, add_noise=False, image_load_path=args.image_load_path_test, image_save_path=args.image_save_path_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    logging.info('num_train_files: ' + str(len(train_dataset.filepaths)))
    logging.info('num_val_files: ' + str(len(val_dataset.filepaths)))

    train_loss = getattr(module_loss, "elr_loss")(num_examp=len(train_dataset.filepaths),
                                                  num_classes=n_classes, beta=args.beta, lambda_=args.lambda_)
    val_loss = getattr(module_loss, "cross_entropy")

    trainer = ModelNetTrainer(model=cnet, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, loss_fn=nn.CrossEntropyLoss(),
                              model_name='svcnn', train_criterion=train_loss, val_criterion=val_loss, num_views=1, writer=writer, eval_step=args.eval_step,
                              log_step=args.logging_step, device=args.device, epochs=args.epochs, monitor=args.monitor, early_stop=args.early_stop, save_dir=save_dir)
    trainer.train()
    time_diff = time.time() - time_stage1_start
    logging.info(f'Train stage_1 used {time_diff} s.')
    return cnet


def train_stage2():
    time_stage2_start = time.time()

    writer = LogWriter(log_path=args.img_log_dir, log_name=args.img_log_name +
                       args.name + str(args.noise_ratio) + "_stage_2")
    # STAGE 2
    save_dir = args.save_dir + '/stage_2'
    cnet = torch.load(os.path.join(args.save_dir + '/stage_1/model_best', "model.pt"))
    cnet_2 = MVCNN(args.name, cnet, nclasses=n_classes, cnn_name=args.cnn_name,
                   num_views=args.num_views, classnames=classnames)
    del cnet
    optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    train_dataset = ImgDataset(args.train_path, classnames=classnames, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views, add_noise=True,
                               noise_ratio=args.noise_ratio, image_load_path=args.image_load_path_train, image_save_path=args.image_save_path_train)
    val_dataset = ImgDataset(args.val_path, classnames=classnames, scale_aug=False, rot_aug=False, test_mode=True, num_models=0,
                             num_views=args.num_views, add_noise=False, image_load_path=args.image_load_path_test_multi, image_save_path=args.image_save_path_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
    logging.info('num_train_files: ' + str(len(train_dataset.filepaths)))
    logging.info('num_val_files: ' + str(len(val_dataset.filepaths)))

    train_loss = getattr(module_loss, "elr_loss")(num_examp=len(train_dataset.filepaths),
                                                  num_classes=n_classes, beta=args.beta, lambda_=args.lambda_)
    val_loss = getattr(module_loss, "cross_entropy")

    trainer = ModelNetTrainer(model=cnet_2, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, loss_fn=nn.CrossEntropyLoss(),
                              model_name='mvcnn', train_criterion=train_loss, val_criterion=val_loss, num_views=args.num_views, writer=writer, eval_step=args.eval_step,
                              log_step=args.logging_step, device=args.device, epochs=args.epochs, monitor=args.monitor, early_stop=args.early_stop, save_dir=save_dir)
    trainer.train()
    time_diff = time.time() - time_stage2_start
    logging.info(f'Train stage_2 used {time_diff} s.')


def train_gvcnn():
    time_gvcnn_start = time.time()
    writer = LogWriter(log_path=args.img_log_dir, log_name=args.img_log_name + args.name + str(args.noise_ratio))
    save_dir = args.save_dir

    cnet = GVCNN(args.name, nclasses=n_classes, pretraining=pretraining, cnn_name=args.cnn_name, group_num=8)
    optimizer = optim.Adam(cnet.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    train_dataset = ImgDataset(args.train_path, classnames=classnames, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views, add_noise=True,
                               noise_ratio=args.noise_ratio, image_load_path=args.image_load_path_train, image_save_path=args.image_save_path_train)
    val_dataset = ImgDataset(args.val_path, classnames=classnames, scale_aug=False, rot_aug=False, test_mode=True, num_models=0,
                             num_views=args.num_views, add_noise=False, image_load_path=args.image_load_path_test, image_save_path=args.image_save_path_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=8)
    logging.info('num_train_files: ' + str(len(train_dataset.filepaths)))
    logging.info('num_val_files: ' + str(len(val_dataset.filepaths)))

    train_loss = getattr(module_loss, "elr_loss")(num_examp=len(train_dataset.filepaths),
                                                  num_classes=n_classes, beta=args.beta, lambda_=args.lambda_)
    val_loss = getattr(module_loss, "cross_entropy")

    trainer = ModelNetTrainer(model=cnet, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, loss_fn=nn.CrossEntropyLoss(),
                              model_name='gvcnn', train_criterion=train_loss, val_criterion=val_loss, num_views=args.num_views, writer=writer, eval_step=args.eval_step,
                              log_step=args.logging_step, device=args.device, epochs=args.epochs, monitor=args.monitor, early_stop=args.early_stop, save_dir=save_dir)
    trainer.train()
    time_diff = time.time() - time_gvcnn_start
    logging.info(f'Train GVCNN used {time_diff} s.')


def train():
    time_start = time.time()
    train_stage1()
    train_stage2()
    # train_gvcnn()
    time_diff = time.time() - time_start
    logging.info(f'Train totally used {time_diff} s.')


if __name__ == '__main__':
    classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                  'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                  'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                  'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                  'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
    n_classes = len(classnames)
    pretraining = not args.no_pretraining
    n_models_train = args.num_models * args.num_views
    train()

# python -m mvcnn_elr.train.train
