import torch
import torchvision.models as models
import numpy as np
from ptflops import get_model_complexity_info
import sys
from models.supermodel import cifarsupermodel26
from models.resnet import resnet18
from utils.tools import model_summary, get_forward_size
from torchscan import summary
from torchstat import stat
from fvcore.nn import FlopCountAnalysis, flop_count

input_size = (3, 32, 32)
model = cifarsupermodel26()
model = models.resnet34(num_classes=100)
# summary = model_summary(model, input_size)
summary = model_summary(model, input_size)
print(get_forward_size(model, input_size))
# stat(model, input_size)
# flops = FlopCountAnalysis(model, torch.rand(1, 3, 32, 32))
# print(flops.total())
sys.exit(0)
print("----------------------------------------------------------------")
line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
print(line_new)
print("================================================================")
total_params = 0
total_output = 0
trainable_params = 0
for layer in summary:
    # input_shape, output_shape, trainable, nb_params
    line_new = "{:>20}  {:>25} {:>15}".format(
        layer,
        str(summary[layer]["output_shape"]),
        "{0:,}".format(summary[layer]["nb_params"]),
    )
    total_params += summary[layer]["nb_params"]
    total_output += np.prod(summary[layer]["output_shape"])
    if "trainable" in summary[layer]:
        if summary[layer]["trainable"] == True:
            trainable_params += summary[layer]["nb_params"]
    print(line_new)

# assume 4 bytes/number (float on cuda).
total_input_size = abs(np.prod(input_size) * 1 * 4. / (1024 ** 2.))
total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
total_size = total_params_size + total_output_size + total_input_size

print("================================================================")
print("Total params: {0:,}".format(total_params))
print("Trainable params: {0:,}".format(trainable_params))
print("Non-trainable params: {0:,}".format(total_params - trainable_params))
print("----------------------------------------------------------------")
print("Input size (MB): %0.2f" % total_input_size)
print("Forward/backward pass size (MB): %0.2f" % total_output_size)
print("Params size (MB): %0.2f" % total_params_size)
print("Estimated Total Size (MB): %0.2f" % total_size)
print("----------------------------------------------------------------")