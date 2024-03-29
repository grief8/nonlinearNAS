import os
import sys
from argparse import ArgumentParser

from utils.putils import generate_arch, get_nas_network
from utils.tools import *
from nas.proxylessnas import _get_module_with_type
from exp.cifar_resnet import resnet18_in, LearnableAlpha
from utils.config import hardware


def analyze_arch(args, hardware):
    from nni.retiarii.fixed import fixed_arch
    with fixed_arch(args.exported_arch_path, verbose=False):
        model = get_nas_network(args)
        print('relu count: ', get_relu_count(model, args.input_size[1:], device='cpu'))
        print('latency: ', predict_latency(model, hardware, args.input_size[1:], device='cpu'))
        throughput, stages = predict_throughput(model, hardware, args.input_size[1:], device='cpu')
        # stages.remove(0.0)
        print('throughput: ', throughput)
        print('stages: ', stages)
        print(max(stages), min(stages), sum(stages) / len(stages))
        print('-----------')


def analyze_relu_count(args, supermodel=False):
    if not supermodel:
        from nni.retiarii.fixed import fixed_arch
        with fixed_arch(args.exported_arch_path, verbose=False):
            model = get_nas_network(args)
            print(get_relu_count(model, args.input_size[1:], device='cpu'))
            print()
    else:
        model = get_nas_network(args)
        print(get_relu_count(model, args.input_size[1:], device='cpu'))
        print()


def get_snl_prediction(model, input_size, ops=None):
    # hardware = {'ReLU': 3.0, 'Conv2d': 0.5, 'AvgPool2d': 3.0, 'BatchNorm2d': 0.05, 'Linear': 0.4, 'MaxPool2d': 3.0, 
    #             'communication': 2.0, 'LayerChoice': 0., 'LearnableAlpha': 3.0}
    if ops is None:
        ops = ['ReLU', 'MaxPool', 'LearnableAlpha']
    summary = model_summary(model, input_size)
    non_ops = _get_module_with_type(model, [LearnableAlpha], [])
    relu_count = []
    for op in non_ops:
        boolean_list = op.alphas.data > 1e-2
        relu_count.append((boolean_list == 1).sum().item())
    # print(summary)
    total, linear = 0.0, 0.0
    stages = []
    idx_alpha = 0 
    for layer in summary:
        nonlinear_flag = False
        key = layer.split('-')[0]
        if not key in hardware.keys():
            continue
        for op in ops:
            if layer.find(op) != -1:
                nonlinear_flag = True
                # print('hit')
                break
        if nonlinear_flag and linear > 0:
            portion = 1
            if key == 'LearnableAlpha':
                portion = relu_count[idx_alpha] / abs(reduce(lambda x, y: x * y, summary[layer]["output_shape"]))
                idx_alpha += 1
            total += max(
                (hardware['communication'] + hardware[key]) * portion * size2memory(
                    summary[layer]["output_shape"]),
                linear)
            stages.append(linear)
            stages.append((hardware['communication'] + hardware[key]) * portion *
                          size2memory(summary[layer]["output_shape"]))
            linear = 0.0
        else:
            linear += hardware[key] * size2memory(summary[layer]["output_shape"])
    total += linear
    if linear > 0.0:
        stages.append(linear)
    return total, sum(stages)


if __name__ == '__main__':
    # model = ['mobilenet', 'resnet18', 'vgg16', 'resnet50']
    # strategy = ['latency', 'throughput']
    # lossType = ['add#linear', 'mul#log']
    # hardware = {'ReLU': 3.0, 'Conv2d': 0.5, 'AvgPool2d': 3.0, 'BatchNorm2d': 0.05, 'Linear': 0.4, 'MaxPool2d': 3.0, 
    #         'communication': 2.0, 'LayerChoice': 0., 'LearnableAlpha': 3.0}

    parser = ArgumentParser("analyzer")
    parser.add_argument('--net', default='vgg16', type=str, help='net type')
    parser.add_argument("--pretrained", default=False, action="store_true")
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset type',
                        choices=['imagenet', 'cifar100'])
    parser.add_argument("--strategy", default='latency', type=str, choices=['latency', 'throughput'])
    parser.add_argument('--grad_reg_loss_type', default='add#linear', type=str,
                        choices=['add#linear', 'mul#log', 'raw'])
    parser.add_argument("--exported_arch_path", default='', type=str)
    parser.add_argument("--checkpoint_path", default='./checkpoints/resnet18/search_net.pt', type=str)
    parser.add_argument("--input_size", default=(1, 3, 224, 224), type=str)
    parser.add_argument('--stride', type=int, default=1, help='conv1 stride')
    parser.add_argument("--choice", default='', type=str)

    args = parser.parse_args()

    if args.choice == 'ours':
        # ours
        from models.supermodel import supermodel16
        model = supermodel16()

        print('----imagenet-----')
        print(get_relu_count(model, (3, 224, 224)))
        print(predict_latency(model, hardware, (3, 224, 224)))
        print(predict_throughput(model, hardware, (3, 224, 224)))
        # print('----cifar-100-----')
        # print(get_relu_count(model, (3, 32, 32)))
        # print(predict_latency(model, hardware, (3, 32, 32)))
        # print(predict_throughput(model, hardware, (3, 32, 32)))
        sys.exit(0)
    elif args.choice == 'SNL':
        # SNL
        model = resnet18_in(num_classes=100, args=args)
        print(args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        # print(get_relu_count(model, (3, 32, 32), ops=['Alpha']))
        # print(eval(args.input_size))
        print(get_snl_prediction(model=model, input_size=eval(args.input_size)[1:]))
        exit(0)
    elif args.choice == 'v0.7':
        base_path = '/scorpio/home/lifabing/projects/nonlinearNAS/checkpoints/oneshot/'
        if args.dataset == 'cifar100':
            args.input_size = (1, 3, 32, 32)
            arch_path = base_path + '{}/{}/{}/checkpoint2.json'.format(args.net, args.strategy, args.grad_reg_loss_type)
        else:
            arch_path = base_path + '{}/{}/{}/checkpoint.json'.format(args.net, args.strategy, args.grad_reg_loss_type)
        if not os.path.exists(arch_path):
            if not os.path.exists(arch_path+'.prob'):
                sys.exit(0)
            generate_arch(arch_path+'.prob')
            arch_path += '.tmp'
        print(args.net, args.dataset, args.strategy, args.grad_reg_loss_type)
        args.exported_arch_path = arch_path
        try:
            analyze_arch(args, hardware)
        except:
            print('running failed')
    else:
        base_path = '/home/lifabing/projects/nonlinearNAS/checkpoints/oneshot/'
        if args.dataset == 'cifar100':
            args.input_size = (1, 3, 32, 32)
            arch_path = base_path + '{}/{}/{}/checkpoint2.json'.format(args.net, args.strategy, args.grad_reg_loss_type)
        else:
            arch_path = base_path + '{}/{}/{}/checkpoint.json'.format(args.net, args.strategy, args.grad_reg_loss_type)
        # if not os.path.exists(arch_path):
        #     if not os.path.exists(arch_path+'.prob'):
        #         sys.exit(0)
        #     generate_arch(arch_path+'.prob')
        #     arch_path += '.tmp'
        # print(args.net, args.dataset, args.strategy, args.grad_reg_loss_type)
        # args.exported_arch_path = arch_path

        # analyze_relu_count(args)
        print(get_relu_count(get_nas_network(args), args.input_size[1:], device='cpu'))
        # analyze_relu_count(args, supermodel=True)
        # hardware = {'nonlinear': 3.0, 'linear': 0.5, 'communication': 4.0}
        # try:
        #     analyze_arch(args, hardware)
        # except:
        #     print('failed')
