import logging

import torch
from utils.tools import model_summary, size2memory

_logger = logging.getLogger(__name__)


class NonlinearLatencyEstimator:
    def __init__(self, hardware, model, dummy_input=(1, 3, 224, 224), ops=None):
        _logger.info(f'Get latency predictor for applied hardware: {hardware}.')
        self.hardware = hardware
        self.summary = model_summary(model, dummy_input)
        self.block_latency_table = {}
        self.linear_lat = 0.0
        if ops is None:
            self.ops = ['ReLU', 'MaxPool']

        self._get_latency_table()

    def _get_latency_table(self):
        linear_lat = 0.0
        for layer in self.summary: 
            nonlinear_flag = False
            for op in self.ops:
                if layer.find(op) != -1:
                    nonlinear_flag = True
                    break
            if nonlinear_flag:
                self.block_latency_table[layer] = size2memory(self.summary[layer]['output_shape']) * self.harware['nonlinear']
            else:
                self.block_latency_table[layer] = size2memory(self.summary[layer]['output_shape']) * self.harware['linear']
                linear_lat += self.block_latency_table[layer]
        self.linear_lat = linear_lat

    def cal_latency(self, current_architecture_prob):
        lat = self.linear_lat
        for module_name, probs in current_architecture_prob.items():
            assert len(probs) == len(self.block_latency_table[module_name])
            lat += torch.sum(torch.tensor([probs[i] * self.block_latency_table[module_name][str(i)]
                                for i in range(len(probs))]))
        return lat

    def export_latency(self, current_architecture):
        lat = self.linear_lat
        for module_name, selected_module in current_architecture.items():
            lat += self.block_latency_table[module_name][str(selected_module)]
        return lat
