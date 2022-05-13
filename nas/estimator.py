import logging

import torch
import nni.retiarii.nn.pytorch as nn
from utils.tools import model_latency, size2memory

_logger = logging.getLogger(__name__)


def _get_module_with_type(root_module, type_name, modules):
    if modules is None:
        modules = []
    if not isinstance(type_name, (list, tuple)):
        type_name = [type_name]

    def apply(m):
        for name, child in m.named_children():
            for t_name in type_name:
                if isinstance(child, t_name):
                    modules.append(child)
                else:
                    apply(child)

    apply(root_module)
    return modules


class NonlinearLatencyEstimator:
    def __init__(self, hardware, model, dummy_input=(1, 3, 224, 224), target='latency'):
        _logger.info(f'Get latency predictor for applied hardware: {hardware}.')
        self.hardware = hardware
        self.in_size = dummy_input
        self.target = target
        self.block_latency_table, self.total_latency = model_latency(model, dummy_input[1:], hardware)
        self.non_ops = _get_module_with_type(model, nn.PReLU, [])
        self.choices = _get_module_with_type(model, nn.LayerChoice, [])
        self._get_layerchoice_names()
        self.layerchoice_latency = self._get_layerchoice()

    def _get_layerchoice(self):
        layerchoice_latency = []
        idx = 0
        for name in self.block_latency_table.keys():
            if name.startswith('LayerChoice'):
                in_size = [int(i) for i in name.split('_')[1].split('x')]
                latency = []
                for choice in self.choices[idx].choices:
                    _, lat = model_latency(choice, in_size, self.hardware)
                    latency.append(lat)
                layerchoice_latency.append(latency)
                self.total_latency -= sum(latency)
                idx += 1
        return layerchoice_latency

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
