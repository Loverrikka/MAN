import sys
import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
from functools import partial
from modeling.backbones.pos_embed import get_2d_sincos_pos_embed
from mmt import models as mdmmt
import random
import copy
import os.path as osp
from libs.utils.prepare_model import create_vit_model
import pdb

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape,
                                                                                                posemb_new.shape, hight,
                                                                                                width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.trans_type = cfg.MODEL.TRANSFORMER_TYPE
        self.batch_size = cfg.SOLVER.IMS_PER_BATCH
        self.instance_num = cfg.DATALOADER.NUM_INSTANCE
        if 't2t' in cfg.MODEL.TRANSFORMER_TYPE:
            self.in_planes = 512
        if 'edge' in cfg.MODEL.TRANSFORMER_TYPE or cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224':
            self.in_planes = 384
        if '14' in cfg.MODEL.TRANSFORMER_TYPE:
            self.in_planes = 384
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        # No view
        view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        num_classes=num_classes,
                                                        camera=camera_num, view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)

        if pretrain_choice == 'imagenet':
            if cfg.MODEL.VehicleMAE:
                self.base.load_param_neo(model_path)
                print('Loading pretrained vehiclemae model......from {}'.format(model_path))

            else:
                self.base.load_param(model_path)
                print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        cash_x = self.base(x, cam_label=cam_label, view_label=view_label)
        global_feat = cash_x[-1][:, 0]
        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cash_x, cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return cash_x, feat
            else:
                return cash_x, global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

#_______________________________________________________________________________________________________________________________________
    
class RESNET50_MMT_DA(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(RESNET50_MMT_DA, self).__init__()
        self.NI = mdmmt.create(cfg.MMT.arch, num_features=cfg.MMT.features, dropout=cfg.MMT.dropout, num_classes=num_classes)
        self.TI = mdmmt.create(cfg.MMT.arch, num_features=cfg.MMT.features, dropout=cfg.MMT.dropout, num_classes=num_classes)
        self.RGB = mdmmt.create(cfg.MMT.arch, num_features=cfg.MMT.features, dropout=cfg.MMT.dropout, num_classes=num_classes)
        
        self.num_classes = num_classes
        self.target_class = cfg.kmeans_class
        self.source_class = self.num_classes - self.target_class
        self.cfg = cfg
        self.dataset_name = cfg.DATASETS.NAMES
        self.mix_dim = 2048
        self.trans_hidden_num = 512
        self.jigsaw_samples = 128
        self.jigsaw_hidden = 512
        if cfg.modal_sl in ['RGB+NIR', 'RGB+TIR', 'NIR+TIR']:
            self.input_dim = 2 * self.mix_dim
        else:
            self.input_dim = 3 * self.mix_dim
        self.proto_weight = cfg.proto_m

        self.hidden_dims = 512  # 隐藏层的尺寸

        self.classifier_F = nn.Linear(3 * self.mix_dim, self.num_classes, bias=False)

        self.classifier_F.apply(weights_init_classifier)
        self.bottleneck_F = nn.BatchNorm1d(3 * self.mix_dim)

        self.bottleneck_F.bias.requires_grad_(False)
        self.bottleneck_F.apply(weights_init_kaiming)

        if self.source_class <= 0:
            pass
        else:
            # 为源域和目标域创建三个模态的原型
            self.register_buffer("prototypes_src_ori", torch.zeros(self.source_class, 3*self.mix_dim))
            self.register_buffer("prototypes_tar_ori", torch.zeros(self.target_class, 3*self.mix_dim))
            self.register_buffer("prototypes_src_R", torch.zeros(self.source_class, self.mix_dim))
            self.register_buffer("prototypes_tar_R", torch.zeros(self.target_class, self.mix_dim))
            self.register_buffer("prototypes_src_N", torch.zeros(self.source_class, self.mix_dim))
            self.register_buffer("prototypes_tar_N", torch.zeros(self.target_class, self.mix_dim))
            self.register_buffer("prototypes_src_T", torch.zeros(self.source_class, self.mix_dim))
            self.register_buffer("prototypes_tar_T", torch.zeros(self.target_class, self.mix_dim))

        self.classifier = nn.Linear(self.input_dim, self.num_classes, bias=False)

        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.input_dim)

        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        
        self.classifier_R = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        self.classifier_R.apply(weights_init_classifier)

        self.bottleneck_R = nn.BatchNorm1d(self.mix_dim)
        self.bottleneck_R.bias.requires_grad_(False)
        self.bottleneck_R.apply(weights_init_kaiming)

        self.classifier_N = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        self.classifier_N.apply(weights_init_classifier)

        self.bottleneck_N = nn.BatchNorm1d(self.mix_dim)
        self.bottleneck_N.bias.requires_grad_(False)
        self.bottleneck_N.apply(weights_init_kaiming)
        
        self.classifier_T = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        self.classifier_T.apply(weights_init_classifier)

        self.bottleneck_T = nn.BatchNorm1d(self.mix_dim)
        self.bottleneck_T.bias.requires_grad_(False)
        self.bottleneck_T.apply(weights_init_kaiming)
        
   
    def set_prototype_update_weight(self, epoch, epochs, cfg):
        start = cfg.pro_weight_range[0]
        end = cfg.pro_weight_range[1]
        self.proto_weight = 1. * epoch / epochs * (end - start) + start
                 
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))
    
    def load_param_neo(self, trained_path):
        param_dict = torch.load(trained_path,map_location ='cpu')
        model_state_dict = self.state_dict()
        
        for name, param in param_dict.items():
            new_name = name.replace('module.', '')
            if new_name in model_state_dict:
                try:
                    if model_state_dict[new_name].shape == param.shape:
                        model_state_dict[new_name].copy_(param)
                    else:
                        print(f'Skipping {new_name} due to shape mismatch: model {model_state_dict[new_name].shape}, checkpoint {param.shape}')
                except Exception as e:
                    print(f'Error copying parameter {new_name}: {e}')
            else:
                print(f'Skipping {new_name} as it is not in the model state dict')

        print('Loading pretrained model from {}'.format(trained_path))
    
            
    def forward(self, x, label=None, cam_label=None, view_label=None, domain=None,modal_sl=None):
        if self.training:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            
            f_out_NI,  p_out_NI = self.NI(NI)
            f_out_TI,  p_out_TI = self.TI(TI)
            f_out_RGB,  p_out_RGB = self.RGB(RGB)

            ori = torch.cat([f_out_RGB, f_out_NI, f_out_TI], dim=-1)     
            
            # ####################################################################################################
    
            if domain == 0:
                # update momentum prototypes with pseudo labels
                self.prototypes_src_ori = self.prototypes_src_ori.detach()
                self.prototypes_src_R = self.prototypes_src_R.detach()
                self.prototypes_src_N = self.prototypes_src_N.detach()
                self.prototypes_src_T = self.prototypes_src_T.detach()

                # update momentum prototypes with pseudo labels
                for feat_R, feat_N, feat_T, ori_q, label_q in zip(f_out_RGB, f_out_NI, f_out_TI, ori, label):
                    self.prototypes_src_ori[label_q] = self.proto_weight * self.prototypes_src_ori[label_q] + (
                                1 - self.proto_weight) * ori_q # torch.cat([feat_R,feat_N,feat_T],dim=-1)                   
                    self.prototypes_src_R[label_q] = self.proto_weight * self.prototypes_src_R[label_q] + (
                                1 - self.proto_weight) * feat_R # ori_q.view(self.mix_dim, 3).mean(dim=-1) # (feat_N + feat_T)
                    self.prototypes_src_N[label_q] = self.proto_weight * self.prototypes_src_N[label_q] + (
                                1 - self.proto_weight) * feat_N# ori_q.view(self.mix_dim, 3).mean(dim=-1)  # (feat_R + feat_T) 
                    self.prototypes_src_T[label_q] = self.proto_weight * self.prototypes_src_T[label_q] + (
                                1 - self.proto_weight) * feat_T # ori_q.view(self.mix_dim, 3).mean(dim=-1) # (feat_R + feat_N)  
                    
            elif domain == 1:
                # update momentum prototypes with pseudo labels
                self.prototypes_tar_ori = self.prototypes_tar_ori.detach()
                self.prototypes_tar_R = self.prototypes_tar_R.detach()
                self.prototypes_tar_N = self.prototypes_tar_N.detach()
                self.prototypes_tar_T = self.prototypes_tar_T.detach()

                # update momentum prototypes with pseudo labels
                for feat_R, feat_N, feat_T, ori_q, label_q in zip(f_out_RGB, f_out_NI, f_out_TI, ori, label):
                    self.prototypes_tar_ori[label_q-self.source_class] = self.proto_weight * self.prototypes_tar_ori[label_q-self.source_class] + (
                                1 - self.proto_weight) * ori_q  # torch.cat([feat_R,feat_N,feat_T],dim=-1) 
                    self.prototypes_tar_R[label_q-self.source_class] = self.proto_weight * self.prototypes_tar_R[label_q-self.source_class] + (
                                1 - self.proto_weight) * feat_R # ori_q.view(self.mix_dim, 3).mean(dim=-1) # (feat_N + feat_T)
                    self.prototypes_tar_N[label_q-self.source_class] = self.proto_weight * self.prototypes_tar_N[label_q-self.source_class] + (
                                1 - self.proto_weight) * feat_N # ori_q.view(self.mix_dim, 3).mean(dim=-1) # (feat_R + feat_T) 
                    self.prototypes_tar_T[label_q-self.source_class] = self.proto_weight * self.prototypes_tar_T[label_q-self.source_class] + (
                                1 - self.proto_weight) * feat_T # ori_q.view(self.mix_dim, 3).mean(dim=-1) # (feat_R + feat_N)   
                   
            else:
                pass
                
            ori_global = self.bottleneck(ori)
            
            ori_score = self.classifier(ori_global)
            
            return ori_score, ori_global, p_out_RGB, f_out_RGB,  p_out_NI, f_out_NI,  p_out_TI, f_out_TI
        
        else:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']

            f_out_NI = self.NI(NI)
            f_out_TI = self.TI(TI)
            f_out_RGB = self.RGB(RGB)
            
            if modal_sl == 'RGB+NIR': 
                ori = torch.cat([f_out_RGB, f_out_NI], dim=-1)
                ori_global = self.bottleneck(ori)
                ori_global = ori
            elif modal_sl =='RGB+TIR' :
                ori = torch.cat([f_out_RGB, f_out_TI], dim=-1)
                ori_global = self.bottleneck(ori)
                ori_global = ori         
            elif modal_sl == 'NIR+TIR' :
                ori = torch.cat([f_out_NI, f_out_TI], dim=-1)
                ori_global = self.bottleneck(ori)
                ori_global = ori         
            else:
                ori = torch.cat([f_out_RGB, f_out_NI, f_out_TI], dim=-1)

                ori_global = self.bottleneck(ori)

                ori_global = ori

            return ori_global, f_out_RGB, f_out_NI, f_out_TI

__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}

def make_model(cfg, num_class=0, camera_num=0, view_num=0):
    if cfg.MODEL.BASE == 9:
         model = RESNET50_MMT_DA(num_class, cfg, camera_num, view_num, __factory_T_type)
         print('===========Building RESNET50_MMT_DA===========')             
    else:
        print('model chioce error !')
    return model
    