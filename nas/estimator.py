import logging
from collections import OrderedDict

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
        # self.non_ops = _get_module_with_type(model, nn.PReLU, [])
        # self.choices = _get_module_with_type(model, nn.LayerChoice, [])
        self.relu_comm_latency = self._get_relu_comm_table()
        self.layerchoice_latency = self._get_layerchoice(_get_module_with_type(model, nn.LayerChoice, []))
        print('NonlinearLatencyEstimator initialized...')

    def _get_layerchoice(self, choices):
        layerchoice_latency = []
        idx = 0
        for name in self.block_latency_table.keys():
            if name.startswith('LayerChoice'):
                in_size = self.block_latency_table[name]['input_shape']
                latency = []
                for i, choice in enumerate(choices[idx].choices):
                    table, lat = model_latency(choice, in_size[:], self.hardware)
                    latency.append(lat)
                    if i == 0:
                        self.total_latency -= lat
                        continue
                    for key in table.keys():
                        if key.startswith('PReLU'):
                            self.relu_comm_latency.append([(size2memory(table[key]['input_shape']) +
                                                            size2memory(
                                                                table[key]['output_shape'])) *
                                                           self.hardware[
                                                               'communication'],
                                                           table[key]['input_shape'][0]])
                layerchoice_latency.append(latency)
                idx += 1
        return layerchoice_latency

    def _get_relu_comm_table(self):
        relu_comm_latency = []
        for name in self.block_latency_table.keys():
            if name.startswith('PReLU'):
                relu_comm_latency.append([(size2memory(self.block_latency_table[name]['input_shape']) +
                                           size2memory(self.block_latency_table[name]['output_shape'])) * self.hardware[
                                              'communication'], self.block_latency_table[name]['input_shape'][0]])
        return relu_comm_latency

    def _cal_latency(self, cur_arch_prob, relu_count):
        lat = self.total_latency
        for i, prob in enumerate(cur_arch_prob):
            for j, pb in enumerate(prob):
                lat += pb * self.layerchoice_latency[i][j]
        for i, count in enumerate(self.relu_comm_latency):
            lat += (1 - relu_count[i] / count[1]) * count[0]
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

    def cal_expected_latency(self, cur_arch_prob, relu_count):
        if self.target == 'latency':
            lat = self._cal_latency(cur_arch_prob, relu_count)
        else:
            lat = self._cal_throughput_latency()
        return lat
