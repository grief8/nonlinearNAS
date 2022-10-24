import time
import math
from datetime import timedelta
import torch
from torch import nn as nn
from utils.putils import BinaryPReLu
from utils.tools import model_summary, size2memory
from nni.nas.pytorch.utils import AverageMeter


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    logsoftmax = nn.LogSoftmax()
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


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


class Retrain:
    def __init__(self, model, optimizer, device, data_provider, n_epochs, export_path, hardware,
                 target, reduce_portion, grad_reg_loss_type, grad_reg_loss_params=None, search_flag=True):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = data_provider.train
        self.valid_loader = data_provider.valid
        self.test_loader = data_provider.test
        self.n_epochs = n_epochs
        self.criterion = nn.CrossEntropyLoss()
        # change it while training
        self.in_size = (1, 3, 32, 32)
        self.export_path = export_path
        self.hardware = hardware
        # we assume batch_size = 1
        self.summary = model_summary(model, self.in_size[1:])
        self.block_latency_table = {}
        for layer in self.summary:
            name = layer.split('-')[0]
            if self.hardware.get(name) is None:
                self.block_latency_table[layer] = 0
                continue
            self.block_latency_table[layer] = size2memory(self.summary[layer]['output_shape']) * self.hardware[name]
        self.non_ops = _get_module_with_type(self.model, BinaryPReLu, [])
        if not search_flag:
            for module in self.non_ops:
                ones = torch.ones_like(module.weight)
                zeros = torch.zeros_like(module.weight)
                module.weight = torch.nn.Parameter(torch.where(module.weight <= 0.5, zeros, ones), requires_grad=False)

        self.reg_loss_type = grad_reg_loss_type
        self.reg_loss_params = {} if grad_reg_loss_params is None else grad_reg_loss_params
        # binary is initialized as ReLU
        self.target = target
        self.ref_latency = self._cal_total_relu() * reduce_portion

    def _cal_total_relu(self):
        relu_count = .0
        idx = 0
        for layer in self.block_latency_table.keys():
            name = layer.split('-')[0]
            if self.hardware.get(name) is None:
                continue
            if name.find('BinaryPReLu') != -1:
                non = self.summary[layer]['output_shape'][1] * self.summary[layer]['output_shape'][2] * self.summary[layer]['output_shape'][3]
                relu_count += non
                idx += 1
        print('network relu count', relu_count)
        return relu_count

    def _cal_latency(self):
        relu_count = .0
        idx = 0
        for layer in self.block_latency_table.keys():
            name = layer.split('-')[0]
            if self.hardware.get(name) is None:
                continue
            if name.find('BinaryPReLu') != -1:
                non = float(torch.abs(torch.sum(self.non_ops[idx].weight))) * self.summary[layer]['output_shape'][2] * self.summary[layer]['output_shape'][3]
                relu_count += non
                idx += 1
        return relu_count

    def _cal_throughput_latency(self):
        linear = size2memory(self.in_size) * self.hardware['communication']
        idx = 0
        stages = []
        for layer in self.block_latency_table.keys():
            name = layer.split('-')[0]
            if self.hardware.get(name) is None:
                continue
            op = self.block_latency_table[layer]
            if name.find('BinaryPReLu') != -1:
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
        lat = self._cal_latency()
        # if self.target == 'latency':
        #     lat = self._cal_latency()
        # else:
        #     lat = self._cal_throughput_latency()
        return lat

    def run(self):
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        # train
        self.train()
        # validate
        self.validate(is_test=False)
        # test
        self.validate(is_test=True)
        print(self.reg_loss_type, self.export_path)
        print('Expected normal latency: ', self._cal_latency())
        print('Expected throughput latency: ', self._cal_throughput_latency())

    def train_one_epoch(self, adjust_lr_func, train_log_func, label_smoothing=0.1):
        batch_time = AverageMeter('batch_time')
        data_time = AverageMeter('data_time')
        losses = AverageMeter('losses')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        self.model.train()
        end = time.time()
        for i, (images, labels) in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            new_lr = adjust_lr_func(i)
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.model(images)
            if label_smoothing > 0:
                ce_loss = cross_entropy_with_label_smoothing(output, labels, label_smoothing)
            else:
                ce_loss = self.criterion(output, labels)
            expected_latency = self.cal_expected_latency()
            print(expected_latency)
            if self.reg_loss_type == 'mul#log':
                import math
                alpha = self.reg_loss_params.get('alpha', 1)
                beta = self.reg_loss_params.get('beta', 0.6)
                # noinspection PyUnresolvedReferences
                reg_loss = (math.log(expected_latency) / math.log(self.ref_latency)) ** beta
                loss = alpha * ce_loss * reg_loss
            elif self.reg_loss_type == 'add#linear':
                reg_lambda = self.reg_loss_params.get('lambda', 2e-1)
                reg_loss = reg_lambda * (expected_latency - self.ref_latency) / self.ref_latency
                loss = ce_loss + reg_loss
            elif self.reg_loss_type == 'snl':
                reg_lambda = self.reg_loss_params.get('lambda', 2e-1)
                regularization_loss = 0
                for param in self.model.parameters():
                    regularization_loss += torch.sum(abs(param))
                loss = ce_loss + reg_lambda * self.ref_latency / expected_latency * regularization_loss
            elif self.reg_loss_type == 'raw':
                loss = ce_loss
            else:
                raise ValueError(f'Do not support: {self.reg_loss_type}')

            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss, images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            self.model.zero_grad()  # or self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0 or i + 1 == len(self.train_loader):
                batch_log = train_log_func(i, batch_time, data_time, losses, top1, top5, new_lr)
                print(batch_log)
        return top1, top5

    def train(self, validation_frequency=1, regular_frequency=50):
        best_acc = 0
        nBatch = len(self.train_loader)
        # for module in self.non_ops:
        #     module.weight = torch.nn.Parameter(torch.rand(module.weight.size()).to(self.device))

        def train_log_func(epoch_, i, batch_time, data_time, losses, top1, top5, lr):
            batch_log = 'Train [{0}][{1}/{2}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                        'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'. \
                format(epoch_ + 1, i, nBatch - 1,
                       batch_time=batch_time, data_time=data_time, losses=losses, top1=top1)
            batch_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
            batch_log += '\tlr {lr:.5f}'.format(lr=lr)
            return batch_log

        def adjust_learning_rate(n_epochs, optimizer, epoch, batch=0, nBatch=None):
            """ adjust learning of a given optimizer and return the new learning rate """
            # cosine
            T_total = n_epochs * nBatch
            T_cur = epoch * nBatch + batch
            # init_lr = 0.05
            new_lr = 0.5 * 0.05 * (1 + math.cos(math.pi * T_cur / T_total))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            return new_lr

        for epoch in range(self.n_epochs):
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')
            end = time.time()
            train_top1, train_top5 = self.train_one_epoch(
                lambda i: adjust_learning_rate(self.n_epochs, self.optimizer, epoch, i, nBatch),
                lambda i, batch_time, data_time, losses, top1, top5, new_lr:
                train_log_func(epoch, i, batch_time, data_time, losses, top1, top5, new_lr),
            )
            # if (epoch + 1) % regular_frequency == 0:
            #     for module in self.non_ops:
            #         ones = torch.ones_like(module.weight)
            #         zeros = torch.zeros_like(module.weight)
            #         module.weight = torch.nn.Parameter(torch.where(module.weight <= 0.5, zeros, ones))

            time_per_epoch = time.time() - end
            seconds_left = int((self.n_epochs - epoch - 1) * time_per_epoch)
            print('Time per epoch: %s, Est. complete in: %s' % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            if (epoch + 1) % validation_frequency == 0:
                val_loss, val_acc, val_acc5 = self.validate(is_test=False)
                is_best = val_acc > best_acc
                best_acc = max(best_acc, val_acc)
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})'. \
                    format(epoch + 1, self.n_epochs, val_loss, val_acc, best_acc)
                val_log += '\ttop-5 acc {0:.3f}\tTrain top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}'. \
                    format(val_acc5, top1=train_top1, top5=train_top5)
                print(val_log)
            else:
                is_best = False
            if is_best:
                torch.save(self.model.module.state_dict(), self.export_path)
                # torch.onnx.export(self.model.module, (torch.rand(self.in_size)).to(self.device), self.export_path)

    def validate(self, is_test=True):
        if is_test:
            data_loader = self.test_loader
        else:
            data_loader = self.valid_loader
        self.model.eval()
        batch_time = AverageMeter('batch_time')
        losses = AverageMeter('losses')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')

        end = time.time()
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                # compute output
                output = self.model(images)
                loss = self.criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0 or i + 1 == len(data_loader):
                    if is_test:
                        prefix = 'Test'
                    else:
                        prefix = 'Valid'
                    test_log = prefix + ': [{0}/{1}]\t' \
                                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                                        'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'. \
                        format(i, len(data_loader) - 1, batch_time=batch_time, loss=losses, top1=top1)
                    test_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
                    print(test_log)
        return losses.avg, top1.avg, top5.avg
