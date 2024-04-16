import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import time
import logging


class ModelNetTrainer(object):

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn,
                 model_name, save_dir, num_views=12, eval_step=None, logging_step=None, writer=None, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.save_dir = save_dir
        self.num_views = num_views
        self.eval_step = eval_step
        self.logging_step = logging_step
        self.writer = writer
        self.device = device

        self.global_step = 0
        self.tic_train = time.time()
        self.loss_list = []
        self.acc_list = []

        self.model.cuda()

    def train(self, n_epochs):

        best_acc = 0
        self.model.train()
        for epoch in tqdm(range(1, n_epochs+1), desc="total progress: "):
            # permute data for mvcnn
            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths)/self.num_views))
            filepaths_new = []
            file_class_new = []
            file_image_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(
                    self.train_loader.dataset.filepaths[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
                file_class_new.extend(
                    self.train_loader.dataset.file_class[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
                file_image_new.extend(
                    self.train_loader.dataset.file_image[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new
            self.train_loader.dataset.file_class = file_class_new
            self.train_loader.dataset.file_image = file_image_new

            # train one epoch
            out_data = None
            in_data = None
            for i, data in enumerate(tqdm(self.train_loader, desc=f"train epoch {epoch}")):

                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                self.writer.add_scalar('params/lr', lr, self.global_step)

                if self.model_name == 'mvcnn':
                    N, V, C, H, W = data[1].size()
                    data[0] = data[0].H[0]
                    in_data = Variable(data[1]).view(-1, C, H, W).cuda()
                else:
                    in_data = Variable(data[1].cuda())
                target = Variable(data[0]).cuda().long()
                self.optimizer.zero_grad()
                out_data = self.model(in_data)
                loss = self.loss_fn(out_data, target)
                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())
                acc = correct_points.float()/results.size()[0]
                loss.backward()
                self.optimizer.step()

                self.loss_list.append(float(loss.cpu()))
                self.acc_list.append(float(acc.cpu()))

                if self.global_step % self.logging_step == 0:
                    time_diff = time.time() - self.tic_train
                    loss_avg = sum(self.loss_list) / len(self.loss_list)
                    acc_avg = sum(self.acc_list) / len(self.acc_list)
                    self.writer.add_scalar("train/loss", loss_avg, self.global_step)
                    self.writer.add_scalar("train/acc", acc_avg, self.global_step)
                    logging.info("global step %d, epoch: %d, loss: %.5f, acc: %.5f, speed: %.2f step/s"
                                 % (self.global_step, epoch, loss_avg, acc_avg, self.logging_step / time_diff))
                    self.tic_train = time.time()

                if self.global_step % self.eval_step == 0:
                    loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy(epoch)
                    self.writer.add_scalar('val/loss', loss, self.global_step)
                    self.writer.add_scalar('val/overall_acc', val_overall_acc, self.global_step)
                    self.writer.add_scalar('val/mean_class_acc', val_mean_class_acc, self.global_step)
                    self.writer.record()

                    # save best model
                    if val_overall_acc > best_acc:
                        logging.info(
                            f"best performence has been updated: {best_acc:.5f} --> {val_overall_acc:.5f}")
                        best_acc = val_overall_acc
                        cur_save_dir = os.path.join(self.save_dir, "model_best")
                        if not os.path.exists(cur_save_dir):
                            os.makedirs(cur_save_dir)
                        torch.save(self.model, os.path.join(cur_save_dir, "model.pt"))

                self.global_step += 1

            # adjust learning rate manually
            if epoch % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.5

    def update_validation_accuracy(self, epoch):
        all_correct_points = 0
        all_points = 0

        # in_data = None
        # out_data = None
        # target = None

        wrong_class = np.zeros(40)
        samples_class = np.zeros(40)
        all_loss = 0

        self.model.eval()

        for _, data in enumerate(tqdm(self.val_loader, desc=f"valid step {self.global_step}: ")):

            if self.model_name == 'mvcnn':
                N, V, C, H, W = data[1].size()
                data[0] = data[0].H[0]
                in_data = Variable(data[1]).view(-1, C, H, W).cuda()
            else:  # 'svcnn'
                in_data = Variable(data[1]).cuda()
            target = Variable(data[0]).cuda()

            out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        logging.info(f'Total # of test models: {all_points}')
        val_mean_class_acc = np.mean((samples_class-wrong_class)/samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        self.model.train()

        return loss, val_overall_acc, val_mean_class_acc
