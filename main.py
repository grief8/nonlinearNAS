import json
import logging
import os
import sys
from argparse import ArgumentParser
import pickle

import torch
from torchvision import transforms
from nni.retiarii.fixed import fixed_arch

import utils.datasets as datasets
from utils.putils import LabelSmoothingLoss, accuracy, get_parameters, get_nas_network
from nas.retrain import Retrain

logger = logging.getLogger('nni_proxylessnas')

if __name__ == "__main__":
    parser = ArgumentParser("proxylessnas")
    # configurations of the model
    parser.add_argument('--net', default='vgg16', type=str, help='net type')
    parser.add_argument("--worker_id", default='0', type=str)
    parser.add_argument("--pretrained", default=False, action="store_true")
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--log_frequency", default=10, type=int)

    parser.add_argument("--no_decay_keys", default='bn', type=str, choices=[None, 'bn', 'bn#bias'])
    parser.add_argument('--grad_reg_loss_type', default='add#linear', type=str, choices=['add#linear', 'mul#log'])
    parser.add_argument('--grad_reg_loss_lambda', default=1e-1, type=float)  # grad_reg_loss_params
    parser.add_argument('--grad_reg_loss_alpha', default=0.2, type=float)  # grad_reg_loss_params
    parser.add_argument('--grad_reg_loss_beta',  default=0.3, type=float)  # grad_reg_loss_params
    parser.add_argument("--applied_hardware", default=None, type=str, help='the hardware to predict model latency')
    parser.add_argument("--reference_latency", default=None, type=float, help='the reference latency in specified hardware')
    # configurations of imagenet dataset
    parser.add_argument("--data_path", default='/home/lifabing/data/imagenet/', type=str)
    parser.add_argument("--train_batch_size", default=48, type=int)
    parser.add_argument("--test_batch_size", default=1024, type=int)
    parser.add_argument("--n_worker", default=32, type=int)
    parser.add_argument("--resize_scale", default=0.08, type=float)
    parser.add_argument("--distort_color", default='normal', type=str, choices=['normal', 'strong', 'None'])
    # configurations for training mode
    parser.add_argument("--train_mode", default='search', type=str, choices=['search', 'retrain'])
    # configurations for search
    parser.add_argument("--checkpoint_path", default='./checkpoints/resnet18/search_net.pt', type=str)
    parser.add_argument("--no-warmup", dest='warmup', action='store_false')
    parser.add_argument("--strategy", default='latency', type=str, choices=['latency', 'throughput'])
    # configurations for retrain
    parser.add_argument("--exported_arch_path", default='./checkpoints/resnet18/checkpoint.json', type=str)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.worker_id
    # torch.cuda.set_device(args.worker_id)

    if args.train_mode == 'retrain' and args.exported_arch_path is None:
        logger.error('When --train_mode is retrain, --exported_arch_path must be specified.')
        sys.exit(-1)

    if args.train_mode == 'retrain':
        assert os.path.isfile(args.exported_arch_path), \
            "exported_arch_path {} should be a file.".format(args.exported_arch_path)
        with fixed_arch(args.exported_arch_path):
            model = get_nas_network(args)
    else:
        model = get_nas_network(args)

    # move network to GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    logger.info('Creating data provider...')
    data_provider = datasets.ImagenetDataProvider(save_path=args.data_path,
                                                  train_batch_size=args.train_batch_size,
                                                  test_batch_size=args.test_batch_size,
                                                  valid_size=None,
                                                  n_worker=args.n_worker,
                                                  resize_scale=args.resize_scale,
                                                  distort_color=args.distort_color)
    logger.info('Creating data provider done')

    if args.no_decay_keys:
        keys = args.no_decay_keys
        momentum, nesterov = 0.9, True
        optimizer = torch.optim.SGD([
            {'params': get_parameters(model, keys, mode='exclude'), 'weight_decay': 4e-5},
            {'params': get_parameters(model, keys, mode='include'), 'weight_decay': 0},
        ], lr=0.05, momentum=momentum, nesterov=nesterov)
    else:
        momentum, nesterov = 0.9, True
        optimizer = torch.optim.SGD(get_parameters(model), lr=0.05, momentum=momentum, nesterov=nesterov, weight_decay=4e-5)

    if args.grad_reg_loss_type == 'add#linear':
        grad_reg_loss_params = {'lambda': args.grad_reg_loss_lambda}
    elif args.grad_reg_loss_type == 'mul#log':
        grad_reg_loss_params = {
            'alpha': args.grad_reg_loss_alpha,
            'beta': args.grad_reg_loss_beta,
        }
    else:
        args.grad_reg_loss_params = None

    if args.train_mode == 'search':
        from nas.proxylessnas import ProxylessTrainer
        from torchvision.datasets import ImageNet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        dataset = ImageNet(args.data_path, transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
        trainer = ProxylessTrainer(model,
                                   loss=LabelSmoothingLoss(),
                                   dataset=dataset,
                                   optimizer=optimizer,
                                   metrics=lambda output, target: accuracy(output, target, topk=(1, 5,)),
                                   num_epochs=args.epochs,
                                   batch_size=args.train_batch_size,
                                   log_frequency=args.log_frequency,
                                   grad_reg_loss_type=args.grad_reg_loss_type, 
                                   grad_reg_loss_params=grad_reg_loss_params, 
                                   applied_hardware=args.applied_hardware, dummy_input=(1, 3, 224, 224),
                                   checkpoint_path=args.exported_arch_path,
                                   strategy=args.strategy)
        trainer.fit()
        print('Final architecture:', trainer.export())
        json.dump(trainer.export(), open(args.exported_arch_path, 'w'))
        json.dump(trainer.export_prob(), open(args.exported_arch_path + '.prob', 'w'))
    elif args.train_mode == 'retrain':
        # this is retrain
        print('this is retrain')
        trainer = Retrain(model, optimizer, device, data_provider, n_epochs=300,
                          export_path=args.exported_arch_path.rstrip('.json') + '.onnx')
        trainer.run()
