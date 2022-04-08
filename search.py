# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
from nni.retiarii import fixed_arch
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils.tools import get_clean_summary
from nas.new_darts import DartsTrainer
from utils import accuracy, get_training_dataloader, get_test_dataloader, get_nas_network, get_network

logger = logging.getLogger('nni')

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument('--net', default='resnet18', type=str, help='net type')
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument("--worker-id", default=0, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument('--loss-type', type=str, default='origin', help='the way to change loss function')
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--constraints", default=1.0, type=float)
    parser.add_argument("--unrolled", default=False, action="store_true")
    parser.add_argument("--visualization", default=True, action="store_true")
    parser.add_argument("--pretrained", default=False, action="store_true")
    parser.add_argument("--arc-checkpoint", default='./checkpoints/oneshot/checkpoint.json', type=str)
    parser.add_argument("--model-path", default="./checkpoints/oneshot/checkpoint.onnx", type=str)

    args = parser.parse_args()

    dataset_train = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True,
        wrap=False
    )
    dataset_valid = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True,
        wrap=False
    )
    torch.cuda.set_device(args.worker_id)
    model = get_nas_network(args)

    criterion = nn.CrossEntropyLoss()

    optim = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)

    summary = get_clean_summary(get_network(args), (3, 32, 32))

    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, 'search', args.net, args.loss_type, args.arc_checkpoint.split('/')[-1].strip('.json')))

    trainer = DartsTrainer(
        model=model,
        loss=criterion,
        metrics=lambda output, target: accuracy(output, target, topk=(1,)),
        optimizer=optim,
        num_epochs=args.epochs,
        dataset=dataset_train,
        batch_size=args.batch_size,
        constraints=args.constraints,
        log_frequency=args.log_frequency,
        unrolled=args.unrolled,
        nonlinear_summary=summary,
        loss_type=args.loss_type,
        writer=writer
    )
    trainer.fit()
    final_architecture = trainer.export()
    print('Final architecture:', trainer.export())
    json.dump(trainer.export(), open(args.arc_checkpoint, 'w'))
    writer.close()