import torch
import torch.nn as nn

from functools import reduce
from collections import OrderedDict

from nni.retiarii.fixed import fixed_arch
from utils.config import nonlinear_ops


def size2memory(size):
    ln = reduce(lambda x, y: x * y, size)
    return abs(ln * 4 / 1024 / 1024)


def sift(summary, ops):
    total_output = 0.0
    for layer in summary:
        for op in ops:
            if layer.find(op) != -1:
                total_output += size2memory(summary[layer]["output_shape"])
    return total_output


def repr_shape(shape):
    if isinstance(shape, (list, tuple)):
        return 'x'.join(str(_) for _ in shape)
    elif isinstance(shape, str):
        return shape
    else:
        return TypeError


def model_latency(model, input_size, hardware, batch_size=1, device="cuda"):
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            if not class_name in list(hardware.keys()):
                return

            input_shape = list(input[0].size())[1:]
            if isinstance(output, (list, tuple)):
                output_shape = [
                    list(o.size())[1:] for o in output
                ]
            else:
                output_shape = list(output.size())[1:]
            module_idx = len(summary)
            m_key = "%s_%s_%s_%i" % (
                class_name, repr_shape(input_shape), repr_shape(output_shape), module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = input_shape
            summary[m_key]["output_shape"] = output_shape
            summary[m_key]['latency'] = hardware[class_name] * size2memory(summary[m_key]["output_shape"]) * batch_size
            total_latency.append(summary[m_key]['latency'])

        hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        model.to(device)
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, (tuple, list)):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    total_latency = []
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return summary, sum(total_latency)


def model_summary(model, input_size, batch_size=-1, device="cuda"):
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            if isinstance(input[0], list):
                input = input[0]
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        model.to(device)
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return summary


def stat_output_data(model, input_size, batch_size=-1, device="cuda", ops=None):
    if ops is None:
        ops = nonlinear_ops
    return sift(model_summary(model, input_size, batch_size, device), ops)


def get_clean_summary(model, input_size, batch_size=-1, device="cuda", ops=None):
    if ops is None:
        ops = nonlinear_ops
    summary = model_summary(model, input_size, batch_size, device)
    new_sum = []
    for layer in summary:
        for op in ops:
            if layer.find(op) != -1:
                new_sum.append(size2memory(summary[layer]["output_shape"]))
    return new_sum


def reconstruct_model(super_model, arc_checkpoints, device="cuda"):
    with fixed_arch(arc_checkpoints, verbose=False):
        model = super_model()
        if device == "cuda" and torch.cuda.is_available():
            model.to(device)
        return model


def predict_latency(model, hardware, input_size, batch_size=-1, device="cuda", ops=None):
    """
    model: nn.Module
        the pytorch model for statistics
    hardware: dict
        the config of hardware platform. e.g. hardware = {'ReLU': 3.0, 'Conv2d': 0.5, 'AvgPool2d': 3.0, 'BatchNorm2d': 0.05, 'Linear': 0.4, 'MaxPool2d': 3.0, 
                'communication': 2.0, 'LayerChoice': 0.0}

    return: latency (ms) per image
    """
    if ops is None:
        ops = nonlinear_ops
    summary = model_summary(model, input_size, batch_size, device)
    total = 0.0
    for layer in summary:
        nonlinear_flag = False
        key = layer.split('-')[0]
        if not key in hardware.keys():
            continue
        for op in ops:
            if layer.find(op) != -1:
                nonlinear_flag = True
                break
        if nonlinear_flag:
            total += (hardware['communication'] + hardware[key]) * size2memory(summary[layer]["output_shape"])
        else:
            total += hardware[key] * size2memory(summary[layer]["output_shape"])
    return total


def predict_throughput(model, hardware, input_size, batch_size=-1, device="cuda", ops=None):
    """
    model: nn.Module
        the pytorch model for statistics
    hardware: dict
        the config of hardware platform. e.g. hardware = {'ReLU': 3.0, 'Conv2d': 0.5, 'AvgPool2d': 3.0, 'BatchNorm2d': 0.05, 'Linear': 0.4, 'MaxPool2d': 3.0, 
                'communication': 2.0, 'LayerChoice': 0.0}

    return: images per second
    """
    if ops is None:
        ops = nonlinear_ops
    summary = model_summary(model, input_size, batch_size, device)
    total, linear = 0.0, 0.0
    stages = []
    for layer in summary:
        nonlinear_flag = False
        key = layer.split('-')[0]
        if not key in hardware.keys():
            continue
        for op in ops:
            if layer.find(op) != -1:
                nonlinear_flag = True
                break
        if nonlinear_flag and linear > 0:
            total += max(
                (hardware['communication'] + hardware[key]) * size2memory(summary[layer]["output_shape"]),
                linear)
            stages.append(linear)
            stages.append((hardware['communication'] + hardware[key]) *
                          size2memory(summary[layer]["output_shape"]))
            linear = 0.0
        else:
            linear += hardware[key] * size2memory(summary[layer]["output_shape"])
    total += linear
    if linear > 0.0:
        stages.append(linear)
    return total, stages


def get_relu_count(model, input_size, batch_size=-1, device="cuda", ops=None):
    if ops is None:
        ops = ['ReLU', 'PReLU', 'Hardswish']
    summary = model_summary(model, input_size, batch_size, device)
    total = 0.0
    for layer in summary:
        nonlinear_flag = False
        for op in ops:
            if layer.find(op) != -1:
                nonlinear_flag = True
                break
        if nonlinear_flag:
            total += abs(reduce(lambda x, y: x * y, summary[layer]["output_shape"]))
    return total


def generate_arrays(y, a, b, value=0, index=0, cur=[]):
    """
    递归生成所有符合条件的数组
    """
    if index == len(y):
        # print(cur)
        if a <= value <= b:
            yield (value, cur)
        return

    # 选择将当前位置设为0
    if value <= b:
        yield from generate_arrays(y, a, b, value, index + 1, cur + [0])

    # 选择将当前位置设为1
    if value <= b:
        yield from generate_arrays(y, a, b, value + y[index], index + 1, cur + [1])


def get_count():
    from models.supermodel import cifarsupermodel50
    from utils.tools import model_summary
    sy = model_summary(cifarsupermodel50(), (3,32,32))
    i = 0
    count_list = []
    for key in sy.keys():
        if 'Identity' in key:
            # print(sy[key]['input_shape'])
            if i % 2 == 0:
                count_list.append(abs(reduce(lambda x, y: x * y, sy[key]['input_shape'])))
            i+=1
    print(count_list)
    a = 90000 # 价值下限
    b = 150000 # 价值上限

    filtered_arrays = [x for x in generate_arrays(count_list, a, b)] 

    print(filtered_arrays) # 打印符合条件的数组