import torch.nn.functional as F
import torch
import torch.nn as nn


def cross_entropy(output, target):
    target = target.long()
    return F.cross_entropy(output, target)


class elr_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=0.3, lambda_=3):
        super(elr_loss, self).__init__()
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp,
                                                                                                        self.num_classes)
        self.lambda_ = lambda_
        self.beta = beta

    def forward(self, index, output, label, num_views, noise_info, true_class):
        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        if num_views != 1:
            true_class = true_class.H[0]
            noise_info = noise_info.H[0]
            for i, idx in enumerate(index):
                self.target[idx:idx + num_views] = self.beta * self.target[idx:idx + num_views] + (1 - self.beta) * (
                    (y_pred_) / (y_pred_).sum(dim=1, keepdim=True))[i, 0]
        else:
            self.target[index] = self.beta * self.target[index] + (1 - self.beta) * (
                (y_pred_) / (y_pred_).sum(dim=1, keepdim=True))

        y_label = torch.eye(self.num_classes).cuda()[label]
        elr_grad = y_pred_ - y_label + self.lambda_ * (y_pred_ / (1 - (self.target[index] * y_pred_).sum(dim=1).unsqueeze(1).expand(
            y_pred_.size())) * ((self.target[index] * y_pred_).sum(dim=1).unsqueeze(1).expand(y_pred_.size()) - (self.target[index] * y_pred_)))
        elr_sim, elr_wht = self.evaluate(torch.gather(elr_grad, 1, true_class.unsqueeze(1)).squeeze(), noise_info)

        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1 - (self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + self.lambda_ * elr_reg

        return final_loss, elr_sim, elr_wht

    def evaluate(self, weight, info):
        weight = torch.abs(weight)
        similarity = F.cosine_similarity(weight, info, dim=0) - \
            F.cosine_similarity(torch.full_like(weight, weight.mean()), info, dim=0)
        weight_change = (weight * info).sum() - (weight.mean() * info).sum()
        return similarity, weight_change
