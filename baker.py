import os
import sys
from argparse import ArgumentParser

from utils.putils import generate_arch, get_nas_network
from utils.tools import predict_latency, predict_throughput, get_relu_count


def analyze_arch(args, hardware):
    from nni.retiarii import fixed_arch
    with fixed_arch(args.exported_arch_path, verbose=False):
        model = get_nas_network(args)
        # print('predict_latency: ', predict_latency(model, hardware, args.input_size[1:]))
        print(predict_latency(model, hardware, args.input_size[1:], device='cpu'))
        # print('predict_throughput: ', predict_throughput(model, hardware, args.input_size[1:]))
        throughput, stages = predict_throughput(model, hardware, args.input_size[1:], device='cpu')
        # stages.remove(0.0)
        print(throughput)
        print(max(stages), min(stages), sum(stages)/len(stages))
        print()
        # print(predict_throughput(model, hardware, args.input_size[1:]))


def analyze_relu_count(args, supermodel=False):
    if not supermodel:
        from nni.retiarii import fixed_arch
        with fixed_arch(args.exported_arch_path, verbose=False):
            model = get_nas_network(args)
            print(get_relu_count(model, args.input_size[1:], device='cpu'))
            print()
    else:
        model = get_nas_network(args)
        print(get_relu_count(model, args.input_size[1:], device='cpu'))
        print()



if __name__ == '__main__':
    # model = ['mobilenet', 'resnet18', 'vgg16', 'resnet50']
    # strategy = ['latency', 'throughput']
    # lossType = ['add#linear', 'mul#log']
    parser = ArgumentParser("analyzer")
    parser.add_argument('--net', default='vgg16', type=str, help='net type')
    parser.add_argument("--pretrained", default=False, action="store_true")
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset type',
                        choices=['imagenet', 'cifar100'])
    parser.add_argument("--strategy", default='latency', type=str, choices=['latency', 'throughput'])
    parser.add_argument('--grad_reg_loss_type', default='add#linear', type=str,
                        choices=['add#linear', 'mul#log', 'raw'])
    parser.add_argument("--exported_arch_path", default='', type=str)
    parser.add_argument("--input_size", default=(1, 3, 224, 224), type=tuple)
    args = parser.parse_args()

    base_path = '/home/lifabing/projects/nonlinearNAS/checkpoints/oneshot/'
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

    analyze_relu_count(args)
    # hardware = {'nonlinear': 3.0, 'linear': 0.5, 'communication': 4.0}
    # try:
    #     analyze_arch(args, hardware)
    # except:
    #     print('failed')