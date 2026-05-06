import torch
from collections import OrderedDict
import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
        
def extract_features(model, data_loader, print_freq=100, logger = None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, pids, cid, vid, fnames) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_vit_feature(model, imgs)
            for index, fname in enumerate(fnames):
                features[fname] = [x[index] for x in outputs]

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0) or ((i + 1) % len(data_loader) == 0) :
                logger.info('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features



def extract_vit_feature(model, inputs):
    inputs['RGB'] = to_torch(inputs['RGB']).cuda()
    inputs['NI'] = to_torch(inputs['NI']).cuda()
    inputs['TI'] = to_torch(inputs['TI']).cuda()
    outputs = model(inputs)
    outputs = [x.data.cpu() for x in outputs]
    return outputs


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

