# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .triplet_loss import  SoftTripletLoss, TripletLoss, TripletLossXBM, MMS_loss, MMDLoss,multiModalMarginLossUnsupervised, multiModalMarginLossNew,Unsupervised_prototype_multiModalMarginLoss,multiModalMarginLossUnsupervised_neo
from .center_loss import CenterLoss
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy, ReverseCrossEntropyLoss
