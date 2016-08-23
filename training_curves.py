#!/usr/bin/env python

## based on https://github.com/dmlc/mxnet/issues/1302
## Parses the model fit log file and generates a train/val vs epoch plot
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse

def log_train_metric(period, auto_reset=False):
    """Callback to log the training evaluation result every period.

    Parameters
    ----------
    period : int
        The number of batch to log the training evaluation metric.
    auto_reset : bool
        Reset the metric after each log

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_epoch_callback to fit.
    """
    def _callback(param):
        """The checkpoint function."""
        if param.nbatch % period == 0 and param.eval_metric is not None:
            name_value = param.eval_metric.get_name_value()
            for name, value in name_value:
                logging.info('Iter[%d] Batch[%d] Train-%s=%f',
                             param.epoch, param.nbatch, name, value)
            if auto_reset:
                param.eval_metric.reset()
    return _callback

parser = argparse.ArgumentParser(description='Parses log file and generates train/val curves')
parser.add_argument('--log-file', type=str,default="log_tr_va",
                    help='the path of log file')
args = parser.parse_args()

print('ok')


TR_RE = re.compile('\s+Train-dicecoef=([\d\.]+)')
VA_RE = re.compile('.*?]\sValidation-dicecoef=([\d\.]+)')

log = open(args.log_file).read()

log_tr = [float(x) for x in TR_RE.findall(log)]
log_va = [float(x) for x in VA_RE.findall(log)]
idx = np.arange(len(log_tr))

print(len(log_tr), len(log_va))


plt.figure(figsize=(8, 6))
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(idx, log_tr, 'o', linestyle='-', color="r",
         label="Train dicecoef")

plt.plot(idx, log_va, 'o', linestyle='-', color="b",
         label="Validation dicecoef")

plt.legend(loc="best")
plt.xticks(np.arange(min(idx), max(idx)+1, 5))
plt.yticks(np.arange(0, 1, 0.2))
plt.ylim([0,1])
plt.show()
