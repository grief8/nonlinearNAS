import logging

import torch
import nni.retiarii.nn.pytorch as nn
from utils.tools import model_latency, size2memory

_logger = logging.getLogger(__name__)


def _get_module_with_type(root_module, type_name, modules):
    if modules is None:
        modules = []

    def apply(m):
        for name, child in m.named_children():
            if isinstance(child, type_name):
                modules.append(child)
            else:
                apply(child)

    apply(root_module)
    return modules


class NonlinearLatencyEstimator:
    def __init__(self, hardware, model, dummy_input=(1, 3, 224, 224), target='latency', strategy='latency'):
        _logger.info(f'Get latency predictor for applied hardware: {hardware}.')
        self.hardware = hardware
        self.strategy = strategy
        self.in_size = dummy_input
        self.target = target
        self.block_latency_table = model_latency(model, dummy_input[1:], hardware)
        self.non_ops = _get_module_with_type(model, nn.PReLU, [])
        self.choices = _get_module_with_type(model, nn.LayerChoice, [])

    def _cal_latency(self):
        lat = .0
        idx = 0
        for layer in self.block_latency_table.keys():
            name = layer.split('-')[0]
            if self.hardware.get(name) is None:
                continue
            op = self.block_latency_table[layer]
            if name.find('PReLu') != -1:
                non = float(torch.sum(self.non_ops[idx].weight))
                lat += op * self.hardware[name]
                lat += op * self.hardware['communication'] * (1 - non / self.summary[layer]['output_shape'][1])
                idx += 1
            else:
                lat += op * self.hardware[name]
        return lat

    def _cal_throughput_latency(self):
        linear = size2memory(self.in_size) * self.hardware['communication']
        idx = 0
        stages = []
        for layer in self.block_latency_table.keys():
            name = layer.split('-')[0]
            if self.hardware.get(name) is None:
                continue
            op = self.block_latency_table[layer]
            if name.find('PReLu') != -1:
                non = float(torch.sum(self.non_ops[idx].weight))
                nonlinear = op * self.hardware[name]
                linear += op * self.hardware['communication'] * (1 - non / self.summary[layer]['output_shape'][1])
                stages.extend([linear, nonlinear])
                linear = op * self.hardware['communication'] * (1 - non / self.summary[layer]['output_shape'][1])
                idx += 1
            else:
                linear += op * self.hardware[name]
        stages.append(linear)
        stages.append(stages[0])
        lat = 0.0
        for i in range(len(stages) - 1):
            lat += max(stages[i], stages[i + 1])
        return lat / 2

    def cal_expected_latency(self):
        if self.target == 'latency':
            lat = self._cal_latency()
        else:
            lat = self._cal_throughput_latency()
        return lat
