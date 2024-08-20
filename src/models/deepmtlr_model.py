from typing import Any, List
import pandas as pd

import torch
from torch import nn
from torchmtlr import MTLR

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, CyclicLR

import torch.nn as nn
import torch.nn.functional as F
from torchmtlr import mtlr_neg_log_likelihood, mtlr_survival, mtlr_risk
import numpy as np
from scipy.spatial import cKDTree

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold

from src.models.components.net import UNETR, ViT
from src.models.unetrpp.unetrpp import UNETR_PP

from torch.utils.tensorboard import SummaryWriter

from src.models.loss.B_losses import BoundaryLoss
from src.models.loss.boundaryloss_utils import calculate_alpha, distance_map_array_gen
import sys


class DEEP_MTLR(LightningModule):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        
        self.validation_step_outputs = []
        self.test_step_outputs = []

        if self.hparams['model'] == 'UNETR':    # if ignore
            self.model = UNETR_PP(hparams=self.hparams,
                                  in_channels=1,
                                  out_channels=1,
                                  feature_size=16,
                                  num_heads=4,
                                  depths=[3, 3, 3, 3],
                                  dims=[32, 64, 128, 256],
                                  do_ds=True,
                                  )
            # self.model.load_state_dict(
            #     torch.load('/home/sribd/Desktop/TMSS_EC_Sorted/model_final_checkpoint.model',
            #              #  map_location='cuda'
            #                ), strict=False)
            self.init_params(self.model)


            # checkpoint = torch.load('/home/sribd/Desktop/TMSS_EC_Sorted/model_final_checkpoint.model')
            # from collections import OrderedDict
            #
            # new_state_dict = OrderedDict()
            # for k, v in checkpoint["state_dict"].items():
            #     if k in ['unetr_pp_encoder.downsample_layers.0.0.conv.weight', 'decoder2.transp_conv.conv.weight',
            #              'out1.conv.conv.weight', 'out1.conv.conv.bias', 'out2.conv.conv.weight', 'out2.conv.conv.bias',
            #              'out3.conv.conv.weight', 'out3.conv.conv.bias']:
            #         continue
            #     # print(k)
            #     new_state_dict[k] = v
            # self.model.load_state_dict(new_state_dict, strict=False)


            # sys.exit()
            # self.model = UNETR(hparams=self.hparams,
            #                    in_channels=1,
            #                    out_channels=1,
            #                    img_size=(160, 160, 64),
            #                    feature_size=16,
            #                    hidden_size=768,
            #                    mlp_dim=3072,
            #                    num_heads=12,
            #                    pos_embed="conv",
            #                    norm_name="instance",
            #                    conv_block=True,
            #                    res_block=True,
            #                    dropout_rate=0.0,
            #                    spatial_dims=3,
            #                    )

        else:
            print('Please select the correct model architecture name.')

        self.loss_mask = Dice_and_FocalLoss()
        self.writer = SummaryWriter("logs")
        self.train_step_count = 0
        self.val_step_count = 0

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def init_params(self, m: torch.nn.Module):
        """Initialize the parameters of a module.
        Parameters
        ----------
        m
            The module to initialize.
        Notes
        -----
        Convolutional layer weights are initialized from a normal distribution
        as described in [1]_ in `fan_in` mode. The final layer bias is
        initialized so that the expected predicted probability accounts for
        the class imbalance at initialization.
        References
        ----------
        .. [1] K. He et al. ‘Delving Deep into Rectifiers: Surpassing
           Human-Level Performance on ImageNet Classification’,
           arXiv:1502.01852 [cs], Feb. 2015.
        """
        print('init_params')
        # pass
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=.1)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, -1.5214691)

    def step(self, batch: Any, train: int):
        (sample, clin_var), y, labels = batch   # image table
        pred_mask, logits = self.forward((sample, clin_var))        #  segment  predict
        # print('model.py  step  logits  ', logits)
        # print("pred mask", pred_mask.shape)
        # print("target mask ", sample['target_mask'].shape)

        loss_mask = self.loss_mask(pred_mask[0], sample['target_mask']) + self.loss_mask(pred_mask[1], sample[
            'target_mask']) + self.loss_mask(pred_mask[2], sample['target_mask'])

        # ------------ boundary loss ---------------------
        # dist_map_label: list[Tensor] = distance_map_array_gen(sample['target_mask'].cpu().numpy())
        # pred_logits: Tensor = pred_mask[0]
        # pred_probs: Tensor = F.sigmoid(pred_logits)
        # boundary_loss_fun = BoundaryLoss(idc=[0])
        # boundary_loss = boundary_loss_fun(pred_probs.to(pred_logits.device), dist_map_label.to(pred_logits.device))
        #
        # alpha = 0.001
        # max_epochs = 100
        # alpha_increase = calculate_alpha(self.current_epoch, max_epochs)
        # alpha = alpha + alpha_increase
        # loss_segment = (1 - alpha) * loss_mask + alpha * boundary_loss  # new name:  loss_segment
        # ----------------------------------------

        loss_mtlr = mtlr_neg_log_likelihood(logits, y.float(), self.model, self.hparams['C1'], average=True)
        # loss_mtlr = 0.0

        if train == 0:
            self.writer.add_scalar("train_loss_mask", loss_mask, self.train_step_count)
            self.writer.add_scalar("train_loss_mtlr", loss_mtlr, self.train_step_count)
            self.train_step_count = self.train_step_count + 1
        elif train == 1:
            self.writer.add_scalar("val_loss_mask", loss_mask, self.val_step_count)
            self.writer.add_scalar("val_loss_mtlr", loss_mtlr, self.val_step_count)
            self.val_step_count = self.val_step_count + 1

        if self.train_step_count < 1000:
            loss_gamma = 1
        elif self.train_step_count < 1000:
            loss_gamma = 0.99
        else:
            loss_gamma = 0.99
        # loss_gamma = 0.99
        loss = (1 - loss_gamma) * loss_mtlr + loss_gamma * loss_mask  # Final Loss
        # loss = (1 - loss_gamma) * loss_mtlr + loss_gamma * loss_segment
        # loss = loss_mtlr
        # loss = loss_mask

        # if self.train_step_count < 2000:
        #     loss = loss_mask
        # else :
        #     loss = (1 - loss_gamma) * loss_mtlr + loss_gamma * loss_mask

        return loss, logits, y, labels, pred_mask[0], sample['target_mask']  # pred_mask[0]

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, labels, pred_mask, target_mask = self.step(batch, 0)

        # self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss}

    def on_train_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, y, labels, pred_mask, target_mask = self.step(batch, 1)

        self.validation_step_outputs.append({"loss": loss, "preds": preds, "y": y, "labels": labels, "pred_mask": pred_mask, "target_mask": target_mask})
        # return {"loss": loss, "preds": preds, "y": y, "labels": labels, "pred_mask": pred_mask,
        #         "target_mask": target_mask}


    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        pred_prob = torch.cat([x["preds"] for x in outputs]).cpu()
        y = torch.cat([x["y"] for x in outputs]).cpu()
        pred_mask = torch.cat([x["pred_mask"] for x in outputs])
        target_mask = torch.cat([x["target_mask"] for x in outputs])
        true_time = torch.cat([x["labels"]["time"] for x in outputs]).cpu()
        true_event = torch.cat([x["labels"]["event"] for x in outputs]).cpu()
        pred_risk = mtlr_risk(pred_prob).detach().numpy()
        # print('true_time  ', true_time)
        # print('pred_prob  ', pred_prob)
        # print(true_event)
        ci_event = concordance_index(true_time, -pred_risk, event_observed=true_event)

        log = {"val/loss": loss,
               "val/ci": ci_event,  # precidt
               "val/dice": dice(pred_mask, target_mask),
               }
        print(self.current_epoch, "  val/loss:   ", loss)
        print(self.current_epoch, "  val/ci:   ", ci_event)
        print(self.current_epoch, "  val/dice: ", dice(pred_mask, target_mask))
        print()
        self.log_dict(log)

        logits = torch.cat([x["preds"] for x in outputs]).cpu()
        pred_risk = mtlr_risk(logits).detach().numpy()

        PatientID = [x['labels']['name'] for x in outputs]
        PatientID = sum(PatientID, [])  # inefficient way to flatten a list

        results = pd.DataFrame({'name': PatientID, 'pred_risk': pred_risk})
        results.to_csv('Predictions.csv')
        
        self.validation_step_outputs.clear()

        return {"loss": loss, "CI": ci_event}

    def test_step(self, batch: Any, batch_idx: int):

        loss, preds, y, labels, pred_mask, target_mask = self.step(batch, 2)
        self.test_step_outputs.append({"loss": loss, "preds": preds, "y": y, "labels": labels, "pred_mask": pred_mask, "target_mask": target_mask})
        # return {"loss": loss, "preds": preds, "y": y, "labels": labels, "pred_mask": pred_mask,
        #         "target_mask": target_mask}

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        pred_prob = torch.cat([x["preds"] for x in outputs]).cpu()
        y = torch.cat([x["y"] for x in outputs]).cpu()
        pred_mask = torch.cat([x["pred_mask"] for x in outputs])
        target_mask = torch.cat([x["target_mask"] for x in outputs])

        true_time = torch.cat([x["labels"]["time"] for x in outputs]).cpu()
        true_event = torch.cat([x["labels"]["event"] for x in outputs]).cpu()

        pred_risk = mtlr_risk(pred_prob).detach().numpy()

        ci_event = concordance_index(true_time, -pred_risk, event_observed=true_event)

        log = {"test/loss": loss,
               "test/ci": ci_event,
               "test/dice": dice(pred_mask, target_mask),
               }

        self.log_dict(log)

        logits = torch.cat([x["preds"] for x in outputs]).cpu()
        pred_risk = mtlr_risk(logits).detach().numpy()

        PatientID = [x['labels']['name'] for x in outputs]
        PatientID = sum(PatientID, [])  # inefficient way to flatten a list

        results = pd.DataFrame({'name': PatientID, 'pred_risk': pred_risk})
        results.to_csv('Predictions.csv')

        self.test_step_outputs.clear()
        return {"loss": loss, "CI": ci_event}

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = make_optimizer(AdamW, self.model, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        scheduler = {
            "scheduler": MultiStepLR(optimizer, milestones=[self.hparams.step], gamma=0.1),

            "monitor": "val/loss",
        }
        return [optimizer], [scheduler]


def make_optimizer(opt_cls, model, **kwargs):
    """Creates a PyTorch optimizer for MTLR training."""
    params_dict = dict(model.named_parameters())
    weights = [v for k, v in params_dict.items() if "mtlr" not in k and "bias" not in k]
    biases = [v for k, v in params_dict.items() if "bias" in k]
    mtlr_weights = [v for k, v in params_dict.items() if "mtlr_weight" in k]
    # Don't use weight decay on the biases and MTLR parameters, which have
    # their own separate L2 regularization
    optimizer = opt_cls([
        {"params": weights},
        {"params": biases, "weight_decay": 0.},
        {"params": mtlr_weights, "weight_decay": 0.},
    ], **kwargs)
    return optimizer


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1

    def forward(self, input, target):
        axes = tuple(range(1, input.dim()))
        intersect = (input * target).sum(dim=axes)
        union = torch.pow(input, 2).sum(dim=axes) + torch.pow(target, 2).sum(dim=axes)
        loss = 1 - (2 * intersect + self.smooth) / (union + self.smooth)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-3

    def forward(self, input, target):
        input = input.clamp(self.eps, 1 - self.eps)
        loss = - (target * torch.pow((1 - input), self.gamma) * torch.log(input) +
                  (1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input))
        return loss.mean()


class Dice_and_FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(Dice_and_FocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.dice_loss(input, target) + self.focal_loss(input, target)
        return loss


def dice(input, target):
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()

    intersect = (bin_input * target).sum(dim=axes)

    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
    score = 2 * intersect / (union + 1e-3)

    return score.mean()


def hausdorff_distance(image0, image1):
    """Code copied from s
    https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/set_metrics.py#L7-L54
    for compatibility reason with python 3.6
    """
    a_points = np.transpose(np.nonzero(image0.cpu()))
    b_points = np.transpose(np.nonzero(image1.cpu()))

    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    return max(max(cKDTree(a_points).query(b_points, k=1)[0]),
               max(cKDTree(b_points).query(a_points, k=1)[0]))
