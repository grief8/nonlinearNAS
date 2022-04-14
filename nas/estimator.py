import logging

import torch
from utils.tools import model_summary, size2memory

_logger = logging.getLogger(__name__)


class NonlinearLatencyEstimator:
    def __init__(self, hardware, model, dummy_input=(1, 3, 224, 224), ops=None, strategy='latency'):
        _logger.info(f'Get latency predictor for applied hardware: {hardware}.')
        self.hardware = hardware
        self.summary = model_summary(model, dummy_input[1:])
        self.block_latency_table = {}
        self.linear_lat = 0.0
        self.nonlinear_lat = 0.0
        if ops is None:
            self.ops = ['ReLU', 'MaxPool']
        self.strategy = strategy

        self._get_latency_table()

    def _get_latency_table(self):
        linear_lat = 0.0
        non_lat = 0.0
        index = 1
        for layer in self.summary:
            if layer.find('LayerChoice') != -1:
                continue
            nonlinear_flag = False
            for op in self.ops:
                if layer.find(op) != -1:
                    nonlinear_flag = True
                    break
            if nonlinear_flag:
                name = 'default_{}'.format(index)
                self.block_latency_table[name] = [size2memory(self.summary[layer]['output_shape']) * self.hardware[
                    'nonlinear'], 0]
                non_lat += self.block_latency_table[name][0]
                index += 1
            else:
                self.block_latency_table[layer] = [size2memory(self.summary[layer]['output_shape']) * self.hardware[
                    'linear']]
                linear_lat += self.block_latency_table[layer][0]
        self.linear_lat = linear_lat
        self.nonlinear_lat = non_lat

    def _cal_throughput_latency(self, current_architecture_prob):
        lat, linear_comm, nonlinear = 0.0, 0.0, 0.0
        cur_arch = current_architecture_prob.keys()
        for key in self.block_latency_table.keys():
            if key in cur_arch:
                # skip if the identity block has higher probability
                if torch.argmax(current_architecture_prob[key]) == len(current_architecture_prob[key]) - 1:
                    continue
                comm = self.block_latency_table[key][0] / self.hardware['nonlinear'] * self.hardware['communication'] * 2
                lat += max(linear_comm + comm,
                           self.block_latency_table[key][0])
                linear_comm, nonlinear = 0.0, 0.0
            else:
                linear_comm += self.block_latency_table[key][0]
        return lat

    def _cal_normal_latency(self, current_architecture_prob):
        lat = self.linear_lat
        for module_name, probs in current_architecture_prob.items():
            assert len(probs) == len(self.block_latency_table[module_name])
            lat += torch.sum(torch.tensor([probs[i] * self.block_latency_table[module_name][i]
                                           for i in range(len(probs))]))
        return lat

    def cal_expected_latency(self, current_architecture_prob):
        if self.strategy == 'latency':
            lat = self._cal_normal_latency(current_architecture_prob)
        else:
            lat = self._cal_throughput_latency(current_architecture_prob)
        return lat

    def export_latency(self, current_architecture):
        lat = self.linear_lat
        for module_name, selected_module in current_architecture.items():
            lat += self.block_latency_table[module_name][selected_module]
        return lat
