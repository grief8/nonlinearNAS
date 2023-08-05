import json
import logging
import os
import sys
from argparse import ArgumentParser
import pickle

import torch
from torch import nn
from torchvision import transforms
from nni.retiarii.fixed import fixed_arch

import utils.datasets as datasets
from models.model import SearchMobileNet
from models.shufflenet import ShuffleNetV2OneShot
from nas.estimator import _get_module_with_type, NonlinearLatencyEstimator
from utils.putils import LabelSmoothingLoss, accuracy, get_parameters, get_nas_network, reproduce_model
from nas.retrain import Retrain
from utils.config import hardware

logger = logging.getLogger('nni_proxylessnas')

if __name__ == "__main__":
    parser = ArgumentParser("proxylessnas")
    # configurations of the model
    parser.add_argument('--net', default='vgg16', type=str, help='net type')
    parser.add_argument("--worker_id", default='0', type=str)
    parser.add_argument("--pretrained", default=False, action="store_true")
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--log_frequency", default=10, type=int)

    parser.add_argument("--n_cell_stages", default='4,4,4,4,4,1', type=str)
    parser.add_argument("--stride_stages", default='2,2,2,1,2,1', type=str)
    parser.add_argument("--width_stages", default='24,40,80,96,192,320', type=str)
    parser.add_argument("--bn_momentum", default=0.1, type=float)
    parser.add_argument("--bn_eps", default=1e-3, type=float)
    parser.add_argument("--dropout_rate", default=0, type=float)
    parser.add_argument("--no_decay_keys", default='bn', type=str, choices=[None, 'bn', 'bn#bias'])
    parser.add_argument('--grad_reg_loss_type', default='add#linear', type=str, choices=['add#linear', 'mul#log', 'raw'])
    parser.add_argument('--grad_reg_loss_lambda', default=1e-1, type=float)  # grad_reg_loss_params
    parser.add_argument('--grad_reg_loss_alpha', default=0.2, type=float)  # grad_reg_loss_params
    parser.add_argument('--grad_reg_loss_beta',  default=0.3, type=float)  # grad_reg_loss_params
    parser.add_argument("--applied_hardware", default=None, type=str, help='the hardware to predict model latency')
    parser.add_argument("--reference_latency", default=None, type=float, help='the reference latency in specified hardware')
    # configurations of imagenet dataset
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset type',
                        choices=['imagenet', 'cifar100', 'cifar10', 'tiny'])
    parser.add_argument("--data_path", default='/home/lifabing/data/imagenet/', type=str)
    parser.add_argument("--train_batch_size", default=48, type=int)
    parser.add_argument("--test_batch_size", default=1024, type=int)
    parser.add_argument("--n_worker", default=16, type=int)
    parser.add_argument("--resize_scale", default=0.08, type=float)
    parser.add_argument("--distort_color", default='normal', type=str, choices=['normal', 'strong', 'None'])
    # configurations for training mode
    parser.add_argument("--train_mode", default='search', type=str, choices=['search', 'retrain'])
    # configurations for search
    parser.add_argument("--checkpoint_path", default='./checkpoints/resnet18/search_net.pt', type=str)
    parser.add_argument("--no-warmup", dest='warmup', action='store_false')
    parser.add_argument("--strategy", default='latency', type=str, choices=['latency', 'throughput'])
    parser.add_argument("--threshold", default=0.5, type=float)
    # configurations for retrain
    parser.add_argument("--exported_arch_path", default='./checkpoints/resnet18/checkpoint.json', type=str)
    parser.add_argument("--kd_teacher_path", default=None, type=str)
    # branch pruning
    parser.add_argument("--std_pruning", default=False, action="store_true")
    parser.add_argument("--clamp", default=False, action="store_true")
    parser.add_argument("--branches", default=4, type=int)

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
            # with open(args.exported_arch_path.rstrip('.json') + '.o', 'rb') as f:
            #     state_dict = pickle.load(f)
            #     new_state_dict = {}
            #     for k, v in state_dict.items():
            #         if 'samplelayer' in k and 'alpha' in k:
            #             new_state_dict[k] = v
            #     model.load_state_dict(new_state_dict, strict=False)
            #     print('std_pruning: {}, clamp: {}'.format(args.std_pruning, args.clamp))
            #     from models.supermodel import _SampleLayer
            #     for _, module in model.named_modules():
            #         if isinstance(module, _SampleLayer):
            #             if args.std_pruning:
            #                 module.freeze(args.branches)
            #             else:
            #                 module.reset_clamp(args.clamp)
    else:
        model = get_nas_network(args)

    # move network to GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    logger.info('Creating data provider {}...'.format(args.dataset))
    if args.dataset == 'imagenet':
        data_provider = datasets.ImagenetDataProvider(save_path=args.data_path,
                                                      train_batch_size=args.train_batch_size,
                                                      test_batch_size=args.test_batch_size,
                                                      valid_size=None,
                                                      n_worker=args.n_worker,
                                                      resize_scale=args.resize_scale,
                                                      distort_color=args.distort_color)
    elif args.dataset == 'tiny':
        data_provider = datasets.TinyImagenetDataProvider(save_path=args.data_path,
                                                      train_batch_size=args.train_batch_size,
                                                      test_batch_size=args.test_batch_size,
                                                      valid_size=None,
                                                      n_worker=args.n_worker,
                                                      resize_scale=args.resize_scale,
                                                      distort_color=args.distort_color)
    elif args.dataset == 'cifar100':
        data_provider = datasets.CIFAR100DataProvider(save_path=args.data_path,
                                                      train_batch_size=args.train_batch_size,
                                                      test_batch_size=args.test_batch_size,
                                                      valid_size=None,
                                                      n_worker=args.n_worker,
                                                      resize_scale=args.resize_scale,
                                                      distort_color=args.distort_color)
    elif args.dataset == 'cifar10':
        data_provider = datasets.CIFAR10DataProvider(save_path=args.data_path,
                                                      train_batch_size=args.train_batch_size,
                                                      test_batch_size=args.test_batch_size,
                                                      valid_size=None,
                                                      n_worker=args.n_worker,
                                                      resize_scale=args.resize_scale,
                                                      distort_color=args.distort_color)
    else:
        print('Failed to create data provider !')
        sys.exit(1)
    logger.info('Creating data provider {} done'.format(args.dataset))

    if args.no_decay_keys:
        keys = args.no_decay_keys
        momentum, nesterov = 0.9, True
        optimizer = torch.optim.SGD([
            {'params': get_parameters(model, keys, mode='exclude'), 'weight_decay': 4e-5},
            {'params': get_parameters(model, keys, mode='include'), 'weight_decay': 0},
        ], lr=0.001, momentum=momentum, nesterov=nesterov)
    else:
        momentum, nesterov = 0.9, True
        optimizer = torch.optim.SGD(get_parameters(model), lr=0.001, momentum=momentum, nesterov=nesterov, weight_decay=4e-5)

    if args.grad_reg_loss_type == 'add#linear':
        grad_reg_loss_params = {'lambda': args.grad_reg_loss_lambda}
    elif args.grad_reg_loss_type == 'mul#log':
        grad_reg_loss_params = {
            'alpha': args.grad_reg_loss_alpha,
            'beta': args.grad_reg_loss_beta,
        }
    else:
        args.grad_reg_loss_params = None

    if args.kd_teacher_path is None:
        teacher = None
    else:
        if args.dataset == 'imagenet':
            from models.volo import volo_d2
            teacher = volo_d2()
        elif args.dataset == 'cifar100':
            from models.teacher import resnet152
            teacher = resnet152()
        else:
            print('cannot load KD model with invalid dataset')
            sys.exit(1)
        teacher.load_state_dict(torch.load(args.kd_teacher_path))
    if args.train_mode == 'search':
        # trainer = Retrain(model, optimizer, device, data_provider, n_epochs=args.epochs,
        #                     export_path=args.checkpoint_path,
        #                     teacher=teacher)
        # trainer.run()
        from nas.proxylessnas import ProxylessTrainer
        trainer = ProxylessTrainer(model,
                                   loss=LabelSmoothingLoss(),
                                   dataset=data_provider.train.dataset,
                                   optimizer=optimizer,
                                   metrics=lambda output, target: accuracy(output, target, topk=(1, 5,)),
                                   num_epochs=args.epochs,
                                   batch_size=args.train_batch_size,
                                   arc_learning_rate=1e-5,
                                   warmup_epochs=60,
                                   log_frequency=args.log_frequency,
                                   grad_reg_loss_type=args.grad_reg_loss_type, 
                                   grad_reg_loss_params=grad_reg_loss_params, 
                                   applied_hardware=hardware, dummy_input=(1,)+data_provider.data_shape,
                                   checkpoint_path=args.exported_arch_path,
                                   strategy=args.strategy,
                                   teacher=teacher)
        trainer.fit()
        print('Final architecture:', trainer.export())
        json.dump(trainer.export(), open(args.exported_arch_path, 'w'))
        json.dump(trainer.export_prob(), open(args.exported_arch_path + '.prob', 'w'))
    elif args.train_mode == 'retrain':
        # this is retrain
        print('this is retrain')
        if args.std_pruning:
            export_path = args.exported_arch_path.rstrip('.json') + '-std_pruning-{}.pth'.format(args.branches)
        else:
            export_path = args.exported_arch_path.rstrip('.json') + '-clamp-{}.pth'.format(args.clamp)
        print(export_path)
        trainer = Retrain(model, optimizer, device, data_provider, n_epochs=args.epochs,
                            export_path=export_path,
                            teacher=teacher)
        trainer.run()
