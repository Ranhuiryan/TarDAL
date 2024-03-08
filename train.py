import argparse
import logging
from argparse import Namespace

import wandb

from pipeline.train import Train
from utils.environment_probe import EnvironmentProbe


def parse_args() -> Namespace:
    # args parser
    parser = argparse.ArgumentParser()

    # universal opt
    parser.add_argument('--id', default='20220925-improved-ddcgan_ir-en_add-bn_tamm-reg', help='train process identifier')
    parser.add_argument('--folder', default='data/train/irfissure_tamm-reg/', help='data root path')
    parser.add_argument('--use_data', default='custom', help='data type (default or custom)')
    parser.add_argument('--size', default=224, help='resize image to the specified size')
    parser.add_argument('--cache', default='cache', help='weights cache folder')

    # model opt
    parser.add_argument('--use_model', default='DDcGAN', help='data type (DDcGAN or TarDAL)')

    # TarDAL opt
    parser.add_argument('--depth', default=3, type=int, help='network dense depth')
    parser.add_argument('--dim', default=32, type=int, help='network features dimension')
    parser.add_argument('--mask', default='m1', help='mark index')
    parser.add_argument('--weight', nargs='+', type=float, default=[1, 20, 0.1], help='loss weight')
    parser.add_argument('--adv_weight', nargs='+', type=float, default=[1, 1], help='discriminator balance')

    # checkpoint opt
    parser.add_argument('--epochs', type=int, default=500, help='epoch to train')
    # optimizer opt
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    # dataloader opt
    parser.add_argument('--batch_size', type=int, default=8, help='dataloader batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='dataloader workers number')

    # experimental opt
    parser.add_argument('--debug', action='store_true', help='debug mode (default: off)')

    return parser.parse_args()


if __name__ == '__main__':
    config = parse_args()
    logging.basicConfig(level='INFO')

    # wandb settings
    wandb.login(key='f48e8fb8666439c6c3064e9d65ae091b6b831866')  # enter yourself wandb api key
    runs = wandb.init(
        project=config.use_model,
        entity="ryanhui",  # enter yourself entity
        config=config,
        mode='disabled' if config.debug else 'online',
        name=config.id,
    )
    config = wandb.config

    environment_probe = EnvironmentProbe()
    train_process = Train(environment_probe, config)
    train_process.run()
