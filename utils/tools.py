import torch
import torch.nn as nn

from functools import reduce
from collections import OrderedDict

from nni.retiarii import fixed_arch


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


def model_summary(model, input_size, batch_size=-1, device="cuda"):
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
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
        ops = ['ReLU', 'MaxPool']
    return sift(model_summary(model, input_size, batch_size, device), ops)


def get_clean_summary(model, input_size, batch_size=-1, device="cuda", ops=None):
    if ops is None:
        ops = ['ReLU', 'MaxPool']
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
