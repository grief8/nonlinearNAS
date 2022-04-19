import os
import sys
from argparse import ArgumentParser

from utils.putils import generate_arch, get_nas_network
from utils.tools import predict_latency, predict_throughput


def analyze_arch(args, hardware):
    from nni.retiarii import fixed_arch
    with fixed_arch(args.exported_arch_path):
        model = get_nas_network(args)
        print('predict_latency: ', predict_latency(model, hardware, args.input_size[1:]))
        print('predict_throughput: ', predict_throughput(model, hardware, args.input_size[1:]))


if __name__ == '__main__':
    # model = ['mobilenet', 'resnet18', 'vgg16', 'resnet50']
    # strategy = ['latency', 'throughput']
    # lossType = ['add#linear', 'mul#log']
    parser = ArgumentParser("analyzer")
    parser.add_argument('--net', default='vgg16', type=str, help='net type')
    parser.add_argument("--pretrained", default=False, action="store_true")
    parser.add_argument("--strategy", default='latency', type=str, choices=['latency', 'throughput'])
    parser.add_argument('--grad_reg_loss_type', default='add#linear', type=str,
                        choices=['add#linear', 'mul#log', 'raw'])
    parser.add_argument("--exported_arch_path", default='', type=str)
    parser.add_argument("--input_size", default=(1, 3, 224, 224), type=tuple)
    args = parser.parse_args()

    base_path = '/home/lifabing/projects/nonlinearNAS/checkpoints/oneshot/'
    arch_path = base_path + '{}/{}/{}/checkpoint.json'.format(args.net, args.strategy, args.grad_reg_loss_type)
    if not os.path.exists(arch_path):
        if not os.path.exists(arch_path+'.prob'):
            sys.exit(0)
        generate_arch(arch_path+'.prob')
        arch_path += '.tmp'
    print(args.net, args.strategy, args.grad_reg_loss_type)

    hardware = {'nonlinear': 3.0, 'linear': 0.5, 'communication': 4.0}
    try:
        analyze_arch(args, hardware)
    except:
        print('failed')