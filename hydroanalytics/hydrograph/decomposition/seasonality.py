# python imports
import os
from typing import List

import pandas as pd
# 3rd party imports
import torch
import torch.distributions
import pytorch_lightning as pl


# project imports


class SeasonalityModel(torch.nn.Module):
    def __init__(self, period: float, num_parameters: int = 3, *args, **kwargs):
        """
        Initializes the seasonality model.
        :param period:
        :param num_parameters:
        """
        super(SeasonalityModel, self).__init__()
        self.num_parameters = num_parameters
        self.period = period

        self.seasonality_mean = torch.nn.Parameter(torch.randn(num_parameters * 2))
        self.seasonality_std = torch.nn.Parameter(torch.full((num_parameters * 2,), 0.05))
        self.seasonality = torch.distributions.Normal(self.seasonality_mean, self.seasonality_std)

    def forward(self, x, num_samples: int = 100):
        """
        Forward pass.
        :param x:
        :param num_samples: The number of samples to generate from parameter distributions
        :return:
        """
        # seasonality = torch.distributions.Normal(self.seasonality_mean, self.seasonality_std)
        parameters = self.seasonality.rsample((num_samples,))
        indexes = torch.arange(start=0, end=self.num_parameters, dtype=torch.float32, device=x.device)

        x = x[..., None, None] * indexes

        cos_x = torch.cos(2 * torch.pi * x / self.period)
        sin_x = torch.sin(2 * torch.pi * x / self.period)
        cat_x = torch.cat([cos_x, sin_x], dim=-1)
        x = cat_x * parameters
        x = torch.sum(x, dim=-1)

        x_mean = torch.mean(x, dim=-1)
        x_std = torch.std(x, dim=-1)

        return torch.distributions.Normal(x_mean, x_std)


class SeasonalityModelPL(pl.LightningModule):

    def __init__(self, period: float, *args, **kwargs):
        """
        Initializes the seasonality model.
        :param period:
        :param num_parameters:
        """
        super(SeasonalityModelPL, self).__init__()
        self.period = period
        self.seasonality_model = SeasonalityModel(period=period, *args, **kwargs)

    def forward(self, x, num_samples: int = 100):
        """
        Forward pass.
        :param x:
        :param num_samples: The number of samples to generate from parameter distributions
        :return:
        """
        return self.seasonality_model(x, num_samples)

    def training_step(self, batch, batch_idx):
        """
        Training step.
        :param batch:  The input data with the following columns:
        :param batch_idx: The batch index
        :return: The loss value for the batch of data provided as input to the model
                 During training, the loss is used to update the model parameters
                 using backpropagation and the optimizer algorithm
        """
        x, y = batch
        x = self(x)

        # mean
        mse = torch.nn.functional.mse_loss(x.mean, y)

        # neg loss likelihood
        ngll = -x.log_prob(y).mean()

        # enforce positive values
        pos = torch.nn.functional.relu(-x.mean + 0.1).sum()

        loss = mse + ngll + pos

        self.log_dict(
            {
                'train_mse': mse,
                'train_ngll': ngll,
                'train_pos': pos,
                'train_loss': loss
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self(x)

        # mean
        mse = torch.nn.functional.mse_loss(x.mean, y)

        # neg loss likelihood
        ngll = -x.log_prob(y).mean()

        # enforce positive values
        pos = torch.nn.functional.relu(-x.mean + 0.1).sum()

        loss = mse + ngll + pos

        self.log_dict(
            {
                'train_mse': mse,
                'train_ngll': ngll,
                'train_pos': pos,
                'train_loss': loss
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self(x)

        # mean
        mse = torch.nn.functional.mse_loss(x.mean, y)

        # neg loss likelihood
        ngll = -x.log_prob(y).mean()

        # enforce positive values
        pos = torch.nn.functional.relu(-x.mean + 0.1).sum()

        loss = mse + ngll + pos

        self.log_dict(
            {
                'train_mse': mse,
                'train_ngll': ngll,
                'train_pos': pos,
                'train_loss': loss
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

        return (
            {
                'optimizer': optimizer,
                # 'lr_scheduler': scheduler,
                'monitor': 'val_loss'
            }
        )