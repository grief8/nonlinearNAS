import json

import torch
import nni.retiarii.nn.pytorch as nn

import sys


def get_parameters(model, keys=None, mode='include'):
    if keys is None:
        for name, param in model.named_parameters():
            yield param
    elif mode == 'include':
        for name, param in model.named_parameters():
            flag = False
            for key in keys:
                if key in name:
                    flag = True
                    break
            if flag:
                yield param
    elif mode == 'exclude':
        for name, param in model.named_parameters():
            flag = True
            for key in keys:
                if key in name:
                    flag = False
                    break
            if flag:
                yield param
    else:
        raise ValueError('do not support: %s' % mode)


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def build_activation(act_func, inplace=True, num_parameters=1):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'prelu':
        return nn.PReLU(num_parameters=num_parameters)
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = dict()
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return res


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.softmax(dim=self.dim) + 1e-8
        pred = torch.log(pred)
        num_classes = pred.size(self.dim)
        # pred = torch.where(torch.eq(pred, torch.zeros_like(pred)), torch.full_like(pred, 1.), pred)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def get_nas_network(args, class_flag=False):
    if args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet
    elif args.net == 'searchmobilenet':
        from models.model import SearchMobileNet
        net = SearchMobileNet
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34
    elif args.net == 'resnet18':
        from models.resnet import resnet152
        net = resnet152
    elif args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn
    elif args.net == 'searchshufflenet':
        from models.shufflenet import ShuffleNetV2OneShot
        net = ShuffleNetV2OneShot
    elif args.net == 'searchresnet18':
        from models.searchresnet import resnet18
        net = resnet18
    elif args.net == 'searchresnet34':
        from models.searchresnet import resnet34
        net = resnet34
    elif args.net == 'searchresnet101':
        from models.searchresnet import resnet101
        net = resnet101
    elif args.net == 'searchresnet152':
        from models.searchresnet import resnet152
        net = resnet152
    elif args.net == 'searchdensenet121':
        from models.searchdensenet import densenet121
        net = densenet121
    elif args.net == 'searchdensenet169':
        from models.searchdensenet import densenet169
        net = densenet169
    elif args.net == 'searchdensenet201':
        from models.searchdensenet import densenet201
        net = densenet201
    elif args.net == 'searchdensenet161':
        from models.searchdensenet import densenet161
        net = densenet161
    elif args.net == 'searchcifardensenet121':
        from models.searchdensenet import cifardensenet121
        net = cifardensenet121
    elif args.net == 'searchcifardensenet169':
        from models.searchdensenet import cifardensenet169
        net = cifardensenet169
    elif args.net == 'searchcifardensenet201':
        from models.searchdensenet import cifardensenet201
        net = cifardensenet201
    elif args.net == 'searchcifardensenet161':
        from models.searchdensenet import cifardensenet161
        net = cifardensenet161
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if not class_flag:
        if args.net == 'searchshufflenet':
            if args.dataset == 'cifar100':
                net = net(input_size=32, n_classes=100)
            else:
                net = net()
        else:
            if args.dataset == 'cifar100':
                n_classes = 100
            elif args.dataset == 'cifar10':
                n_classes = 10
            elif args.dataset == 'imagenet':
                n_classes = 1000
            else:
                n_classes = 1000
            net = net(pretrained=args.pretrained, num_classes=n_classes)

    return net


def generate_arch(checkpoint_prob_path, output_path=None):
    if output_path is None:
        output_path = checkpoint_prob_path.rstrip('.prob') + '.tmp'
    with open(checkpoint_prob_path, 'r') as f, open(output_path, 'w') as out:
        prob = json.load(f)
        arch = dict()
        for key in prob.keys():
            arch[key] = prob[key].index(max(prob[key]))
        json.dump(arch, out)
        return arch


class BinaryPReLu(nn.Module):
    def __init__(self):
        super(BinaryPReLu, self).__init__()
        self.relu = nn.PReLU()

    def forward(self, x):
        weight = self.relu.weight
        # torch.clamp(weight, 0, 1)
        ones = torch.ones_like(weight)
        zeros = torch.zeros_like(weight)
        self.relu.weight = torch.where(weight <= 0.5, zeros, ones)
        return self.relu(x)
