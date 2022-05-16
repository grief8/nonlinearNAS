import copy
import logging
import sys
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
        self._refresh_table(_get_module_with_type(model, nn.LayerChoice, []))
        print('NonlinearLatencyEstimator initialized...')

    def _refresh_table(self, choices):
        idx = 0
        new_table = copy.copy(self.block_latency_table)
        for name in self.block_latency_table.keys():
            if name.startswith('LayerChoice'):
                in_size = self.block_latency_table[name]['input_shape']
                layer_table = OrderedDict()
                for i, choice in enumerate(choices[idx].choices):
                    table, lat = model_latency(choice, in_size[:], self.hardware)
                    layer_table[str(i)] = table
                    if i == 0:
                        self.total_latency -= lat
                        for j, key in enumerate(table):
                            rm_key = '_'.join(key.split('_')[:-1] + [str(int(name.split('_')[-1]) + j - 8)])
                            new_table.pop(rm_key)
                        continue
                new_table[name] = layer_table
                idx += 1
        self.block_latency_table = copy.copy(new_table)

    def _cal_latency(self, cur_arch_prob, relu_count):
        total = 0.0
        for name in self.block_latency_table.keys():
            if name.startswith('LayerChoice'):
                layer_choice_prob = cur_arch_prob.pop(0)
                table = self.block_latency_table[name]
                for idx, key in enumerate(table):
                    to = 0.0
                    for _, op in enumerate(table[key]):
                        if op.startswith('PReLU'):
                            relu_idx = relu_count.pop(0)
                            to += (size2memory(table[key][op]['input_shape']) + size2memory(
                                table[key][op]['output_shape'])) \
                                  * self.hardware['communication'] + table[key][op]['latency'] * \
                                  (1 - relu_idx / table[key][op]['input_shape'][0])
                            if relu_idx / table[key][op]['input_shape'][0] > 1:
                                sys.exit(0)
                        else:
                            to += table[key][op]['latency']
                    total += to * layer_choice_prob[idx]
            elif name.startswith('PReLU'):
                relu_idx = relu_count.pop(0)
                total += (size2memory(self.block_latency_table[name]['input_shape']) + size2memory(
                    self.block_latency_table[name]['output_shape'])) \
                         * self.hardware['communication'] + self.block_latency_table[name]['latency'] * \
                         (1 - relu_idx / self.block_latency_table[name]['input_shape'][0])
            else:
                total += self.block_latency_table[name]['latency']
        return total

    def _cal_throughput_latency(self, cur_arch_prob, relu_count):
        sequence = []
        total = linear = 0.0
        for name in self.block_latency_table.keys():
            if name.startswith('LayerChoice'):
                if linear > 0:
                    sequence.append(linear)
                    linear = .0
                layer_choice_prob = cur_arch_prob.pop(0)
                # choose the max prob branch
                key = str(int(layer_choice_prob.max(0).values))
                table = self.block_latency_table[name]
                lin = 0.0
                sub_seq = []
                for _, op in enumerate(table[key]):
                    if op.startswith('PReLU'):
                        if lin > 0:
                            sub_seq.append(lin)
                            lin = 0.0
                        relu_idx = relu_count.pop(0)
                        sub_seq.append((size2memory(table[key][op]['input_shape']) + size2memory(
                            table[key][op]['output_shape'])) \
                                       * self.hardware['communication'] + table[key][op]['latency'] * \
                                       (1 - relu_idx / table[key][op]['input_shape'][0]))

                    else:
                        lin += table[key][op]['latency']
                if lin > 0:
                    sub_seq.append(lin)
                sequence.extend(sub_seq)
            elif name.startswith('PReLU'):
                if linear > 0:
                    sequence.append(linear)
                    linear = .0
                relu_idx = relu_count.pop(0)
                sequence.append((size2memory(self.block_latency_table[name]['input_shape']) + size2memory(
                    self.block_latency_table[name]['output_shape'])) \
                         * self.hardware['communication'] + self.block_latency_table[name]['latency'] * \
                         (1 - relu_idx / self.block_latency_table[name]['input_shape'][0]))
            else:
                linear += self.block_latency_table[name]['latency']
        if linear > 0:
            sequence.append(linear)

        sequence.append(sequence[0])
        for i in range(len(sequence) - 1):
            total += max(sequence[i], sequence[i + 1])
        return total / 2

    def cal_expected_latency(self, cur_arch_prob, relu_count):
        if self.target == 'latency':
            lat = self._cal_latency(cur_arch_prob, relu_count)
        else:
            lat = self._cal_throughput_latency(cur_arch_prob, relu_count)
        return lat
