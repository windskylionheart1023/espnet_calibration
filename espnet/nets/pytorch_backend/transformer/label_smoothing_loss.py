#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Label smoothing module."""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.

    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    """

    def __init__(
        self,
        size,
        padding_idx,
        smoothing,
        normalize_length=False,
        criterion=nn.KLDivLoss(reduction="none"),
    ):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def forward(self, x, target):
        """Compute loss between x and target.

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target:
            target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx  # (B,)
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)  # avoid -1 index
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom

class FocalLabelSmoothingLoss(nn.Module):
    """Focal Label Smoothing Loss.

    :param int size: the number of classes
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means conventional focal CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    :param float gamma: focusing parameter for focal loss
    """

    def __init__(
        self,
        size,
        padding_idx,
        smoothing,
        normalize_length=False,
        criterion=nn.KLDivLoss(reduction="none"),
        gamma=2.0
    ):
        """Construct a FocalLabelSmoothingLoss object."""
        super(FocalLabelSmoothingLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.gamma = gamma
        self.normalize_length = normalize_length

    def forward(self, x, target):
        """Compute Focal Label Smoothing Loss between x and target.

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target: target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)  # Flatten to (batch * seqlen, class)
        target = target.view(-1)   # Flatten to (batch * seqlen,)
        
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx  # Mask padding indices
            total = len(target) - ignore.sum().item()  # Count of non-ignored elements
            target = target.masked_fill(ignore, 0)  # Avoid -1 index in scatter_
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Compute log-probabilities and focal weight
        log_probs = F.log_softmax(x, dim=1)
        probs = torch.exp(log_probs)
        
        # Apply focal weight: (1 - p_t)^gamma
        focal_weight = ((1 - probs) ** self.gamma).detach()
        
        # Calculate the Focal Label Smoothing Loss
        kl = self.criterion(log_probs * focal_weight, true_dist)
        
        # Normalization
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom

class AlphaBetaDivLoss(_Loss):
    def __init__(self, alpha: float, beta: float, reduction: str = 'mean'):
        """
        Initialize the Alpha-Beta Divergence Loss.

        Args:
            alpha (float): Alpha parameter for the divergence. Should not be zero.
            beta (float): Beta parameter for the divergence. Should not be zero.
            reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', 'sum'.
        """
        super(AlphaBetaDivLoss, self).__init__()
        if alpha == 0 or beta == 0:
            raise ValueError("Alpha and Beta must be non-zero for the Alpha-Beta divergence.")
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Alpha-Beta Divergence Loss.

        Args:
            input (torch.Tensor): The log-probabilities (log-softmax) or scores of the model (B x C).
            target (torch.Tensor): The target probability distribution (B x C).

        Returns:
            torch.Tensor: The computed Alpha-Beta Divergence Loss.
        """
        # Ensure the target is a valid probability distribution
        if not torch.allclose(target.sum(dim=-1), torch.tensor(1.0, device=target.device), atol=1e-5):
            raise ValueError("Target should be a valid probability distribution.")

        # Ensure the input is in log-space
        input_probs = input.exp()

        # Compute component terms of the Alpha-Beta Divergence
        term1 = (target ** self.alpha) * (input_probs ** self.beta)
        term2 = (self.alpha / (self.alpha + self.beta)) * (target ** (self.alpha + self.beta))
        term3 = (self.beta / (self.alpha + self.beta)) * (input_probs ** (self.alpha + self.beta))
        # scalar = -1/(self.alpha * self.beta)

        divergence = -1/(self.alpha * self.beta) * (term1 - term2 - term3)
        # divergence = -1 * (term1 - term2 - term3)

        # Apply reduction
        if self.reduction == 'mean':
            return divergence.mean()
        elif self.reduction == 'sum':
            return divergence.sum()
        elif self.reduction == "batchmean":  # mathematically correct
            return divergence.sum() / input.size(0)
        else:  # 'none'
            return divergence