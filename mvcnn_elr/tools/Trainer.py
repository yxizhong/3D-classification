import torch
import numpy as np
import logging
import time
import os
from tqdm import tqdm
from mvcnn_elr.ELR.base.base_trainer import BaseTrainer
from typing import List


class ModelNetTrainer(BaseTrainer):

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, model_name, train_criterion, val_criterion,
                 num_views=12, writer=None, eval_step=None, log_step=None, device=None, epochs=None, monitor=None, early_stop=None, save_dir=None):
        super().__init__(model, train_criterion, optimizer, val_criterion, device, epochs, monitor, early_stop)

        self.model = model
        self.train_loader = train_loader
        self.n_classnames = len(train_loader.dataset.classnames)

        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.num_views = num_views
        self.writer = writer
        self.eval_step = eval_step
        self.save_dir = save_dir
        self.do_validation = self.val_loader is not None
        self.do_test = False
        self.log_step = log_step
        self.train_loss_list: List[float] = []
        self.val_loss_list: List[float] = []

        self.global_step = 0
        self.tic_train = time.time()
        self.loss_list = []
        self.acc_list = []
        self.elr_sim_list = []
        self.elr_wht_list = []
        self.not_improved_count = 0
        self.model.cuda()

    def _train_epoch(self, n_epochs):

        self.model.train()

        rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths)/self.num_views))
        filepaths_new = []
        file_class_new = []
        file_image_new = []
        noise_info_new = []
        true_class_new = []
        for i in range(len(rand_idx)):
            filepaths_new.extend(
                self.train_loader.dataset.filepaths[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            file_class_new.extend(
                self.train_loader.dataset.file_class[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            file_image_new.extend(
                self.train_loader.dataset.file_image[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            noise_info_new.extend(
                self.train_loader.dataset.noise_info[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            true_class_new.extend(
                self.train_loader.dataset.true_class[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
        self.train_loader.dataset.filepaths = filepaths_new
        self.train_loader.dataset.file_class = file_class_new
        self.train_loader.dataset.file_image = file_image_new
        self.train_loader.dataset.noise_info = noise_info_new
        self.train_loader.dataset.true_class = true_class_new

        for batch_idx, (label, data, indexs, noise_info, true_class) in enumerate(tqdm(self.train_loader, desc=f"train epoch {n_epochs}: ")):
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, self.global_step)
            if self.model_name == 'mvcnn':
                N, V, C, H, W = data.size()
                in_data = data.view(-1, C, H, W).cuda()
                label = label.H[0]
                indexs = [self.train_loader.dataset.filepaths.index(filepath) for filepath in indexs[0]]
            elif self.model_name == 'gvcnn':
                in_data = data.cuda()
                label = label.H[0]
                indexs = [self.train_loader.dataset.filepaths.index(filepath) for filepath in indexs[0]]
            else:
                in_data = data.cuda()
                indexs = [self.train_loader.dataset.filepaths.index(filepath) for filepath in indexs]

            target = label.cuda().long()

            out_data = self.model(in_data)

            loss, elr_sim, elr_wht = self.train_criterion(indexs, out_data, target,
                                                          self.num_views, noise_info.cuda(), true_class.cuda())
            pred = torch.max(out_data, 1)[1]
            results = pred == target
            correct_points = torch.sum(results.long())
            train_acc = correct_points.float() / results.size()[0]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_list.append(float(loss.cpu()))
            self.acc_list.append(float(train_acc.cpu()))
            self.elr_sim_list.append(float(elr_sim.cpu()))
            self.elr_wht_list.append(float(elr_wht.cpu()))

            if batch_idx % self.log_step == 0:
                time_diff = time.time() - self.tic_train
                loss_avg = sum(self.loss_list) / len(self.loss_list)
                acc_avg = sum(self.acc_list) / len(self.acc_list)
                elr_sim_avg = sum(self.elr_sim_list) / len(self.elr_sim_list)
                elr_wht_avg = sum(self.elr_wht_list) / len(self.elr_wht_list)
                self.writer.add_scalar("train/loss", loss_avg, self.global_step)
                self.writer.add_scalar("train/acc", acc_avg, self.global_step)
                self.writer.add_scalar("elr/simliarity", elr_sim_avg, self.global_step)
                self.writer.add_scalar("elr/weight_change", elr_wht_avg, self.global_step)
                logging.info("global step %d, epoch: %d, loss: %.5f, acc: %.5f, elr_sim: %.5f, elr_wht: %.5f, speed: %.2f step/s"
                             % (self.global_step, n_epochs, loss_avg, acc_avg, elr_sim_avg, elr_wht_avg, self.log_step / time_diff))
                self.tic_train = time.time()

            if batch_idx % self.eval_step == 0:
                val_loss, val_overall_acc, val_class_acc = self._valid_epoch(n_epochs)
                self.writer.add_scalar("val/loss", val_loss, self.global_step)
                self.writer.add_scalar("val/overall_acc", val_overall_acc, self.global_step)
                self.writer.add_scalar("val/mean_class_acc", val_class_acc, self.global_step)
                self.writer.record()

                logging.info("val_loss: %.5f, val_overall_acc: %.5f, val_class_acc: %.5f" %
                             (val_loss, val_overall_acc, val_class_acc))
                if self.mnt_mode != 'off':
                    try:
                        improved = (self.mnt_mode == 'min' and val_overall_acc <= self.mnt_best) or (
                            self.mnt_mode == 'max' and val_overall_acc >= self.mnt_best)
                    except KeyError:
                        logging.info("Warning: Overall acc '{}' is not found. "
                                     "Model performance monitoring is disabled.".format(self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        logging.info(
                            f"best overall acc performence has been updated: {self.mnt_best:.5f} --> {val_overall_acc:.5f}")
                        self.mnt_best = val_overall_acc
                        self.not_improved_count = 0

                        # 单个模型仅保存最优结果
                        cur_save_dir = os.path.join(self.save_dir, "model_best")
                        if not os.path.exists(cur_save_dir):
                            os.makedirs(cur_save_dir)
                        torch.save(self.model, os.path.join(cur_save_dir, "model.pt"))
                    else:
                        self.not_improved_count += 1

                    if self.not_improved_count > self.early_stop:
                        logging.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                        break
            self.global_step += 1

        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step()

        if n_epochs % 10 == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5

    def _valid_epoch(self, epoch):
        all_correct_points = 0
        all_points = 0

        wrong_class = np.zeros(self.n_classnames)
        samples_class = np.zeros(self.n_classnames)

        self.model.eval()

        total_val_loss = 0
        with torch.no_grad():
            for label, data, _, _, _ in tqdm(self.val_loader, total=len(self.val_loader), desc=f"valid step {self.global_step}: "):
                if self.model_name == 'mvcnn':
                    N, V, C, H, W = data.size()
                    label = label.H[0]
                    in_data = data.view(-1, C, H, W).cuda()
                elif self.model_name == 'gvcnn':
                    label = label.H[0]
                    in_data = data.cuda()
                else:
                    in_data = data.cuda()

                target = label.cuda().long()
                out_data = self.model(in_data)
                loss = self.val_criterion(out_data, target)
                pred = torch.max(out_data, 1)[1]
                results = pred == target

                for i in range(results.size()[0]):
                    if not bool(results[i].cpu().data.numpy()):
                        wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                    samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
                correct_points = torch.sum(results.long())

                all_correct_points += correct_points
                all_points += results.size()[0]

                self.val_loss_list.append(loss.item())
                total_val_loss += loss.item()

        logging.info(f'Total # of test models: {all_points}')
        val_class_acc = np.mean((samples_class - wrong_class) / samples_class)
        val_overall_acc = all_correct_points.float() / all_points
        val_overall_acc = val_overall_acc.cpu().data.numpy()
        val_loss = total_val_loss / len(self.val_loader)
        self.model.train()

        return val_loss, val_overall_acc, val_class_acc
