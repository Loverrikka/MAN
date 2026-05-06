import os
from config import cfg
# os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
from utils.logger import setup_logger
from data import make_dataloader, make_cluster_dataloader
from modeling import make_model
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler
import random
import torch
import numpy as np
import argparse
import math
import time
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP, R1_mAP_eval_allday
from torch.cuda import amp
from layers import CrossEntropyLabelSmooth, TripletLoss, SoftTripletLoss, TripletLossXBM, Unsupervised_prototype_multiModalMarginLoss
from engine.processor import do_inference, do_inference_neo
from utils.metrics import extract_features_neo
from torch_clustering import PyTorchKMeans, PyTorchGaussianMixture, evaluate_clustering
import collections
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from utils.metrics import accuracy
from modeling.backbones.xbm import XBM
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
from torch.distributions import Categorical
from sklearn.metrics.pairwise import rbf_kernel
from collections import defaultdict
from cluster_test import km as ckm
from solver.make_optimizer import make_optimizer
from collections import Counter
import pdb
from modeling.backbones.cm_neo import ClusterMemory

output_dir_map = { 9: 'RESNET50_MMT_DA'}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def filter_layers(stage):
    layer_names = ['conv', 'layer1', 'layer2', 'layer3', 'layer4', 'feat_bn']
    ori_bn_names = []
    idm_bn_names = []
    for i in range(len(layer_names)):
        if i < stage+1:
            ori_bn_names.append(layer_names[i])
        else:
            idm_bn_names.append(layer_names[i])
    return idm_bn_names


def parse_data(inputs,device):
    img, vid, target_cam, target_view, _  = inputs
    img = {'RGB': img['RGB'].to(device),
            'NI': img['NI'].to(device),
            'TI': img['TI'].to(device)}
    target = vid.to(device)
    target_cam = target_cam.to(device)
    target_view = target_view.to(device)

    return img, target, target_cam, target_view

def parse_data_MMT(inputs,device):
    img1, img2, vid, target_cam, target_view, _  = inputs
    img1 = {'RGB': img1['RGB'].to(device),
            'NI': img1['NI'].to(device),
            'TI': img1['TI'].to(device)}
    img2 = {'RGB': img2['RGB'].to(device),
            'NI': img2['NI'].to(device),
            'TI': img2['TI'].to(device)}
    target = vid.to(device)
    target_cam = target_cam.to(device)
    target_view = target_view.to(device)

    return img1, img2, target, target_cam, target_view

class MMDLoss(torch.nn.Module):
    def __init__(self, kernel='gaussian', bandwidth=1.0):
        super(MMDLoss, self).__init__()
        self.kernel = kernel
        self.bandwidth = bandwidth

    def gaussian_kernel(self, x, y):
        # 计算高斯核
        xx = torch.sum(x ** 2, dim=1).view(-1, 1)
        yy = torch.sum(y ** 2, dim=1).view(1, -1)
        dists = xx + yy - 2 * torch.mm(x, y.t())
        return torch.exp(-dists / (2 * self.bandwidth ** 2))

    def forward(self, x, y):
        # 计算两个分布的 MMD
        if self.kernel == 'gaussian':
            Kxx = self.gaussian_kernel(x, x)  # K(x, x)
            Kyy = self.gaussian_kernel(y, y)  # K(y, y)
            Kxy = self.gaussian_kernel(x, y)  # K(x, y)

            # 计算 MMD
            mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
        else:
            raise ValueError("Currently, only Gaussian kernel is implemented.")

        return mmd

#___________________________________________________________
# 计算簇之间的匹配成本
def compute_cost_matrix(clusters1, clusters2, num_clusters):
    cost_matrix = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(num_clusters):
            cost_matrix[i, j] = np.sum((clusters1 == i) != (clusters2 == j))
    return cost_matrix

# 匹配聚类结果
def match_clusters(clusters1, clusters2, num_clusters):
    cost_matrix = compute_cost_matrix(clusters1, clusters2, num_clusters)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

#___________________________________________________________


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="TransReID Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "--model_sl", default=None, help="select_model", type=int
    )
    parser.add_argument(
        "--data_iter", default=False, help="select_model", type=bool
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    if args.model_sl:
        cfg.MODEL.BASE = args.model_sl
    cfg.dataloader_iter = args.data_iter
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)
    
    output_dir = cfg.OUTPUT_DIR
    output_dir = output_dir + cfg.DATASETS.NAMES + '2' + cfg.TARGET_DATASETS.NAMES + '/' + output_dir_map[cfg.MODEL.BASE]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("USPReid", output_dir, cfg, if_train=True)
    logger.info("Saving model in the path :{}".format(output_dir))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    
    dataset, target_dataset, train_loader, source_train_loader, target_train_loader, val_loader, target_val_loader, num_query, target_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    
    true_labels = np.array([data[1] for data in target_dataset.train])
    cam_labels_s = np.array([data[2] for data in dataset.train])
    cam_labelss_t = np.array([data[2] for data in target_dataset.train])
    source_classes = num_classes
    target_classes = cfg.kmeans_class
    num_classes = source_classes+target_classes
 
    if cfg.TARGET_DATASETS.NAMES == 'market_to_RGBNT201' or cfg.TARGET_DATASETS.NAMES == 'RGBNT100':
        dist_type = 'l2'     
    else:
        dist_type = 'cos'
        
    print("data is ready")
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param_neo(cfg.MMT.init_1)
            
    model.cuda()
    model = nn.DataParallel(model)
    
    logger.info("test on target dataset: {}".format(cfg.TARGET_DATASETS.NAMES))
    do_inference(cfg,
                model,
                target_val_loader,
                target_query,
                logger,
                )
    logger.info("test on source dataset: {}".format(cfg.DATASETS.NAMES))
    do_inference_neo(cfg,
                model,
                val_loader,
                num_query,
                logger,)
    
    # Create XBM
    datasetSize = len(dataset.train)+len(target_dataset.train)
    memorySize = int(cfg.idm.ratio*datasetSize)
    if cfg.MODEL.BASE == 10 or cfg.MODEL.BASE ==7:
        xbm = XBM(memorySize, cfg.VIT.featureSize)
    else:
        xbm = XBM(memorySize, cfg.idm.featureSize)
    print('XBM memory size = ', memorySize)
        
    source_features, source_features1, source_features2, source_features3, _ = extract_features_neo(model, source_train_loader, print_freq=50)

    sour_fea_dict = collections.defaultdict(list)
    sour_fea_dict1 = collections.defaultdict(list)
    sour_fea_dict2 = collections.defaultdict(list)
    sour_fea_dict3 = collections.defaultdict(list)

    # 首先收集所有特征
    for f, pid, _, _ in sorted(dataset.train):
        if type(f) == type("This is a str"):
            sour_fea_dict[pid].append(source_features[f].unsqueeze(0))
            sour_fea_dict1[pid].append(source_features1[f].unsqueeze(0))
            sour_fea_dict2[pid].append(source_features2[f].unsqueeze(0))
            sour_fea_dict3[pid].append(source_features3[f].unsqueeze(0))
        else:
            sour_fea_dict[pid].append(source_features[f[0]].unsqueeze(0))
            sour_fea_dict1[pid].append(source_features1[f[0]].unsqueeze(0))
            sour_fea_dict2[pid].append(source_features2[f[0]].unsqueeze(0))
            sour_fea_dict3[pid].append(source_features3[f[0]].unsqueeze(0))

    # 计算类中心和原型
    source_centers = []
    source_prototypes = []
    source_centers1 = []
    source_prototypes1 = []
    source_centers2 = []
    source_prototypes2 = []
    source_centers3 = []
    source_prototypes3 = []

    for pid in sorted(sour_fea_dict.keys()):
        # 处理每个模态的特征
        for modality_idx, (feat_dict, centers_list, prototypes_list) in enumerate(zip(
            [sour_fea_dict, sour_fea_dict1, sour_fea_dict2, sour_fea_dict3],
            [source_centers, source_centers1, source_centers2, source_centers3],
            [source_prototypes, source_prototypes1, source_prototypes2, source_prototypes3]
        )):
            # 获取当前模态的特征
            feats = torch.cat(feat_dict[pid], 0)
            
            # 计算初始中心（均值）
            current_center = feats.mean(dim=0)
            
            # 计算样本到中心的余弦距离
            distances = 1 - F.cosine_similarity(feats, current_center.unsqueeze(0))
            
            # 使用softmax转换距离为权重（距离越小权重越大）
            weights = F.softmax(-distances / 0.1, dim=0)  # 0.1是温度参数，可调整
            
            # 计算加权原型
            weighted_prototype = (feats * weights.view(-1, 1)).sum(dim=0)
            
            # 保存结果
            centers_list.append(current_center)
            prototypes_list.append(weighted_prototype)

    # 转换为tensor并移到GPU
    source_centers = torch.stack(source_centers, 0).cuda()
    source_prototypes = torch.stack(source_prototypes, 0).cuda()
    source_centers1 = torch.stack(source_centers1, 0).cuda()
    source_prototypes1 = torch.stack(source_prototypes1, 0).cuda()
    source_centers2 = torch.stack(source_centers2, 0).cuda()
    source_prototypes2 = torch.stack(source_prototypes2, 0).cuda()
    source_centers3 = torch.stack(source_centers3, 0).cuda()
    source_prototypes3 = torch.stack(source_prototypes3, 0).cuda()
 
    if cfg.TARGET_DATASETS.NAMES == 'RGBNT201':
        pass
    else:
        model.module.classifier.weight.data[0:source_classes].copy_(F.normalize(source_centers, dim=1).cuda())
        model.module.RGB.classifier.weight.data[0:source_classes].copy_(F.normalize(source_centers1, dim=1).cuda())
        model.module.NI.classifier.weight.data[0:source_classes].copy_(F.normalize(source_centers2, dim=1).cuda())
        model.module.TI.classifier.weight.data[0:source_classes].copy_(F.normalize(source_centers3, dim=1).cuda())
   
    model.module.prototypes_src_ori = source_prototypes
    model.module.prototypes_src_R = source_prototypes1
    model.module.prototypes_src_N = source_prototypes2    
    model.module.prototypes_src_T = source_prototypes3    

    optimizer = make_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(args.local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              find_unused_parameters=True)

    loss_meter = AverageMeter()
    losses_ce = AverageMeter()
    losses_tri = AverageMeter()
    losses_xbm = AverageMeter()
    losses_cls = AverageMeter()
    losees_3m = AverageMeter()
    losses_3m_p = AverageMeter()
    losses_mmd = AverageMeter()
    acc_r_source_meter = AverageMeter()
    acc_n_source_meter = AverageMeter()
    acc_t_source_meter = AverageMeter()
    acc_r_target_meter = AverageMeter()
    acc_n_target_meter = AverageMeter()
    acc_t_target_meter = AverageMeter()
    
    if cfg.DATASETS.NAMES  in ["MSVR310", "MSVR310_neo","MSVR310_neo_100"]:
        evaluator1 = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    elif cfg.DATASETS.NAMES == "alldayRNTG" :
        evaluator1 = R1_mAP_eval_allday(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator1 = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    
    if cfg.TARGET_DATASETS.NAMES  in ["MSVR310", "MSVR310_neo","MSVR310_neo_100"]:
        evaluator = R1_mAP(target_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    elif cfg.TARGET_DATASETS.NAMES == "alldayRNTG" :
        evaluator = R1_mAP_eval_allday(target_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(target_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        
    scaler = amp.GradScaler()
    # train
    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}
    best_index1 = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}
    
    for epoch in range(1, epochs + 1):
        
        start_time = time.time()
        
        logger.info('Extract feat and Calculate dist...')
        dict_f, dict_f1, dict_f2, dict_f3, _ = extract_features_neo(model, target_train_loader, print_freq=50)
            
        cf = torch.stack(list(dict_f.values())).numpy()
        cf1 = torch.stack(list(dict_f1.values())).numpy()
        cf2 = torch.stack(list(dict_f2.values())).numpy()
        cf3 = torch.stack(list(dict_f3.values())).numpy()
        cf = torch.tensor(cf, device=device)
        cf1 = torch.tensor(cf1, device=device)
        cf2 = torch.tensor(cf2, device=device)
        cf3 = torch.tensor(cf3, device=device)
            
        cf_list = [cf, cf1, cf2, cf3]

        # 定义参数
        num_clusters = target_classes
        logger.info('\n Clustering into {} classes \n'.format(num_clusters))
        
        km1 = PyTorchKMeans(n_clusters=num_clusters, metric='cosine', random_state=cfg.SOLVER.SEED)
        km2 = PyTorchKMeans(n_clusters=num_clusters, metric='cosine', random_state=cfg.SOLVER.SEED)
        km3 = PyTorchKMeans(n_clusters=num_clusters, metric='cosine', random_state=cfg.SOLVER.SEED)
        km = PyTorchKMeans(n_clusters=num_clusters, metric='cosine', random_state=cfg.SOLVER.SEED)
        
        target_label1 = km1.fit_predict(cf1).cpu().numpy()
        target_label2 = km2.fit_predict(cf2).cpu().numpy()
        target_label3 = km3.fit_predict(cf3).cpu().numpy()
        target_label = km.fit_predict(cf).cpu().numpy()
        pseudo_labels = target_label

        target_centers = torch.tensor(km.cluster_centers_,dtype=torch.float32)
        target_centers_R = torch.tensor(km1.cluster_centers_,dtype=torch.float32)
        target_centers_N = torch.tensor(km2.cluster_centers_,dtype=torch.float32)
        target_centers_T = torch.tensor(km3.cluster_centers_,dtype=torch.float32)

        y_pred_mapped, cluster_acc = ckm(true_labels, target_label)
        logger.info('accuracy of pseudo labels :{0}'.format(cluster_acc))
        y_pred_mapped, cluster_acc = ckm(true_labels, target_label1)
        logger.info('accuracy of pseudo labels :{0}'.format(cluster_acc))
        y_pred_mapped, cluster_acc = ckm(true_labels, target_label2)
        logger.info('accuracy of pseudo labels :{0}'.format(cluster_acc))
        y_pred_mapped, cluster_acc = ckm(true_labels, target_label3)
        logger.info('accuracy of pseudo labels :{0}'.format(cluster_acc))

        # 匈牙利算法匹配
        match12 = match_clusters(target_label, target_label1, num_clusters)
        match13 = match_clusters(target_label, target_label2, num_clusters)
        match14 = match_clusters(target_label, target_label3, num_clusters)

        # 生成伪标签并筛选可靠样本
        pseudo_labels = np.full_like(target_label, -1)
        reliable_indices = []
        
        for i in range(num_clusters):
            # 确定当前簇在各模态中的匹配关系
            matched_cluster2 = match12[1][i]
            matched_cluster3 = match13[1][i]
            matched_cluster4 = match14[1][i]
    
            # 找到在三个模态中的标签匹配情况
            for idx in range(len(target_label)):
                cluster_match_count = 0
                if target_label[idx] == i:
                    if target_label1[idx] == matched_cluster2:
                        cluster_match_count += 1
                    if target_label2[idx] == matched_cluster3:
                        cluster_match_count += 1
                    if target_label3[idx] == matched_cluster4:
                        cluster_match_count += 1

                # 如果聚类标签至少有两个一致，进行分类器一致性判断
                if cluster_match_count >= 2:
                        pseudo_labels[idx] = i
                        if idx not in reliable_indices:
                            reliable_indices.append(idx)

        reliable_indices = np.array(reliable_indices)
        logger.info("Reliable sample number :{0}".format(len(reliable_indices)))

        y_pred_mapped, cluster_acc = ckm(true_labels, pseudo_labels)
        logger.info('accuracy of pseudo labels :{0}'.format(cluster_acc))
        
        # # 将特征矩阵和簇中心放在列表中
        # reliable_centers_list = [
        #     torch.zeros(num_clusters, c.shape[1], dtype=torch.float32).to(c.device) for c in cf_list
        # ]

        # 筛选可靠样本
        reliable_feats_list = [c[reliable_indices] for c in cf_list]  # 获取每个模态的可靠样本
        reliable_true_labels = true_labels[reliable_indices]  # 获取可靠样本的真实标签
        reliable_pseudo_labels = pseudo_labels[reliable_indices]  # 获取可靠样本的伪标签

        # 计算可靠样本的伪标签准确率
        y_pred_mapped, reliable_cluster_acc = ckm(reliable_true_labels, reliable_pseudo_labels)
        logger.info('Accuracy of reliable pseudo labels :{0}'.format(reliable_cluster_acc))

        # # 计算每个簇的中心
        # for i in range(num_clusters):
        #     for feats, centers in zip(reliable_feats_list, reliable_centers_list):
        #         cluster_feats = feats[reliable_pseudo_labels == i]
        #         if len(cluster_feats) > 0:
        #             centers[i] = cluster_feats.mean(dim=0)

        # logger.info(f"Reliable centers calculated for {num_clusters} clusters.")
        
        # 将特征矩阵和簇中心放在列表中
        reliable_centers_list = [
            torch.zeros(num_clusters, c.shape[1], dtype=torch.float32).to(c.device) for c in cf_list
        ]

        # 计算每个簇的加权原型
        for i in range(num_clusters):
            for modality_idx, (feats, centers) in enumerate(zip(reliable_feats_list, reliable_centers_list)):
                # 获取当前模态当前簇的所有样本
                cluster_mask = (reliable_pseudo_labels == i)
                cluster_feats = feats[cluster_mask]
                
                if len(cluster_feats) > 0:
                    # 计算样本到当前原型的距离（使用余弦距离）
                    current_center = cluster_feats.mean(dim=0)  # 初始中心
                    distances = 1 - F.cosine_similarity(cluster_feats, current_center.unsqueeze(0))
                    
                    # 将距离转换为权重（距离越小权重越大）
                    # 使用softmax转换（可以尝试不同的温度参数）
                    weights = F.softmax(-distances / 0.1, dim=0)  # 0.1是温度参数，可调整
                    
                    # 计算加权原型
                    weighted_prototype = (cluster_feats * weights.view(-1, 1)).sum(dim=0)
                    reliable_centers_list[modality_idx][i] = weighted_prototype

        logger.info(f"Weighted reliable centers calculated for {num_clusters} clusters.")

        # MSLE
        #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
        
        # # 定义参数
        # num_clusters = target_classes
        # logger.info('\n Clustering into {} classes \n'.format(num_clusters))
        # km = PyTorchKMeans(n_clusters=num_clusters, random_state=cfg.SOLVER.SEED)
        
        # target_label = km.fit_predict(cf).cpu().numpy()
        # pseudo_labels = target_label

        # target_centers = torch.tensor(km.cluster_centers_,dtype=torch.float32)
        
        # y_pred_mapped, cluster_acc = ckm(true_labels, target_label)
        # logger.info('accuracy of pseudo labels :{0}'.format(cluster_acc))
        
        # B,C = target_centers.size()
        # target_centers_R = target_centers[:, :C//3]
        # target_centers_N = target_centers[:, C//3:2*C//3]
        # target_centers_T = target_centers[:, 2*C//3:]  
        
        #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
        
        # 更新target_dataset中的标签
        for i in range(len(target_dataset.train)):
            target_dataset.train[i] = list(target_dataset.train[i])
            target_dataset.train[i][1] = int(pseudo_labels[i] + source_classes)
            target_dataset.train[i] = tuple(target_dataset.train[i])
        
        # 取出可靠标签索引的图片组成新的new_dataset
        new_dataset = [target_dataset.train[i] for i in reliable_indices]
        # new_dataset = target_dataset.train

        # 使用新的new_dataset创建数据加载器
        train_loader_target = make_cluster_dataloader(cfg, new_dataset)
        #_______________________________________________________________________________            
        

        if (epoch == 1 and (cfg.TARGET_DATASETS.NAMES == 'RGBNT100' )) or cfg.TARGET_DATASETS.NAMES == 'RGBNT201' or cfg.TARGET_DATASETS.NAMES == 'market_to_RGBNT201' : # 
            pass
        else:

            model.module.classifier.weight.data[source_classes:source_classes+target_classes].copy_(F.normalize(target_centers, dim=1).float().cuda())
            model.module.classifier_R.weight.data[source_classes:source_classes+target_classes].copy_(F.normalize(target_centers_R, dim=1).float().cuda())
            model.module.classifier_N.weight.data[source_classes:source_classes+target_classes].copy_(F.normalize( target_centers_N, dim=1).float().cuda())
            model.module.classifier_T.weight.data[source_classes:source_classes+target_classes].copy_(F.normalize(target_centers_T, dim=1).float().cuda())
                

        
        # model.module.prototypes_tar_ori = target_centers
        # model.module.prototypes_tar_R = target_centers_R
        # model.module.prototypes_tar_N = target_centers_N
        # model.module.prototypes_tar_T = target_centers_T
        model.module.prototypes_tar_ori = reliable_feats_list[0]
        model.module.prototypes_tar_R = reliable_feats_list[1]
        model.module.prototypes_tar_N = reliable_feats_list[2]    
        model.module.prototypes_tar_T = reliable_feats_list[3]   
        
        # del target_centers_R, target_centers_N, target_centers_T
        del target_centers, target_centers_R, target_centers_N, target_centers_T, cf, cf1, cf2, cf3

        #--------------------------------------------------------------------------------------------------------
        criterion = nn.CrossEntropyLoss(reduction='none').cuda()
        criterion_ce = CrossEntropyLabelSmooth(source_classes + target_classes).cuda()
        criterion_triple = SoftTripletLoss(margin=0.0).cuda()
        criterion_tri_xbm = TripletLossXBM(margin=0.0).cuda()
        criterion_3m = Unsupervised_prototype_multiModalMarginLoss(margin=0.5,dist_type=dist_type).cuda()

        # 初始化损失函数
        criterion_mmd = MMDLoss(bandwidth=1.0)

        loss_meter.reset()
        losses_ce.reset()
        losses_tri.reset()
        losses_xbm.reset()
        losses_cls.reset()
        losees_3m.reset()
        losses_3m_p.reset()
        losses_mmd.reset()
        acc_r_source_meter.reset()
        acc_n_source_meter.reset()
        acc_t_source_meter.reset()
        acc_r_target_meter.reset()
        acc_n_target_meter.reset()
        acc_t_target_meter.reset()
        evaluator.reset()
        evaluator1.reset()
        scheduler.step(epoch)
        model.train()
        
        train_loader.new_epoch()
        train_loader_target.new_epoch()
        
        for n_iter in range(len(train_loader_target)):
            source_inputs = train_loader.next()
            target_inputs = train_loader_target.next()
            img_source, vid_source, target_cam_source, target_view_source = parse_data(source_inputs,device)
            img_target, vid_target, target_cam_target, target_view_target = parse_data(target_inputs,device)

            img_source = {'RGB': img_source['RGB'].to(device),
                        'NI': img_source['NI'].to(device),
                        'TI': img_source['TI'].to(device)}
            target_source = vid_source.to(device)
            target_cam_source = target_cam_source.to(device) if cfg.MODEL.SIE_CAMERA else None
            target_view_source = target_view_source.to(device) if cfg.MODEL.SIE_VIEW else None

            img_target = {'RGB': img_target['RGB'].to(device),
                        'NI': img_target['NI'].to(device),
                        'TI': img_target['TI'].to(device)}
            target_target = vid_target.to(device)
            target_cam_target = target_cam_target.to(device) if cfg.MODEL.SIE_CAMERA else None
            target_view_target = target_view_target.to(device) if cfg.MODEL.SIE_VIEW else None  

            optimizer.zero_grad()
            with amp.autocast(enabled=True):
                                            
                source_ori_score, source_ori, source_RGB_score, source_RGB_global, source_NI_score, source_NI_global, source_TI_score, source_TI_global = model(img_source, label=target_source, cam_label=target_cam_source, view_label=target_view_source, domain=0)                    
                target_ori_score, target_ori, target_RGB_score, target_RGB_global, target_NI_score, target_NI_global, target_TI_score, target_TI_global = model(img_target, label=target_target, cam_label=target_cam_target, view_label=target_view_target, domain=1)                   
                
                # CE + TRIPLET _______________________________________________________________________________________________________________________________________________
                loss_ce_s = criterion_ce(source_ori_score, target_source) + criterion_ce(source_RGB_score, target_source) + criterion_ce(source_NI_score, target_source) + criterion_ce(source_TI_score, target_source)
                loss_tri_s = criterion_triple(source_ori,source_ori,target_source) + criterion_triple(source_RGB_global,source_RGB_global,target_source) + criterion_triple(source_NI_global,source_NI_global,target_source) + criterion_triple(source_TI_global,source_TI_global,target_source) #Soft triplet
                loss_ce_t =criterion_ce(target_ori_score, target_target) + criterion_ce(target_RGB_score, target_target) + criterion_ce(target_NI_score, target_target) + criterion_ce(target_TI_score, target_target)
                loss_tri_t = criterion_triple(target_ori,target_ori,target_target) + criterion_triple(target_RGB_global,target_RGB_global,target_target) + criterion_triple(target_NI_global,target_NI_global,target_target) + criterion_triple(target_TI_global,target_TI_global,target_target) #Soft triplet

                loss_ce = loss_ce_s + loss_ce_t
                loss_tri = loss_tri_s + loss_tri_t            
                
                # 3M loss _______________________________________________________________________________________________________________________________________________
                loss_3m_1 = criterion_3m(F.normalize(target_RGB_global),F.normalize(target_NI_global),F.normalize(target_TI_global),F.normalize(model.module.prototypes_src_R),F.normalize(model.module.prototypes_src_N),F.normalize(model.module.prototypes_src_T))
                loss_3m_2 = criterion_3m(F.normalize(source_RGB_global),F.normalize(source_NI_global),F.normalize(source_TI_global),F.normalize(model.module.prototypes_tar_R),F.normalize(model.module.prototypes_tar_N),F.normalize(model.module.prototypes_tar_T))
                loss_3m = loss_3m_1 + loss_3m_2

                #_______________________________________________________________________________________________________________________________________________
                loss_mmd_1 = criterion_mmd(F.normalize(model.module.prototypes_src_ori), target_ori) + criterion_mmd(F.normalize(model.module.prototypes_src_R), F.normalize(target_RGB_global)) + criterion_mmd(F.normalize(model.module.prototypes_src_N), F.normalize(target_NI_global)) + criterion_mmd(F.normalize(model.module.prototypes_src_T), F.normalize(target_TI_global)) 
                loss_mmd_2 = criterion_mmd(F.normalize(model.module.prototypes_tar_ori), source_ori) + criterion_mmd(F.normalize(model.module.prototypes_tar_R), F.normalize(source_RGB_global)) + criterion_mmd(F.normalize(model.module.prototypes_tar_N), F.normalize(source_NI_global)) + criterion_mmd(F.normalize(model.module.prototypes_tar_T), F.normalize(source_TI_global)) 
                
                loss_mmd_3 = criterion_mmd(F.normalize(model.module.prototypes_src_R), F.normalize(target_NI_global)) + criterion_mmd(F.normalize(model.module.prototypes_src_N), F.normalize(target_TI_global)) + criterion_mmd(F.normalize(model.module.prototypes_src_T), F.normalize(target_RGB_global)) + \
                                criterion_mmd(F.normalize(model.module.prototypes_src_R), F.normalize(target_TI_global)) + criterion_mmd(F.normalize(model.module.prototypes_src_N), F.normalize(target_RGB_global)) + criterion_mmd(F.normalize(model.module.prototypes_src_T), F.normalize(target_NI_global))
                loss_mmd_4 = criterion_mmd(F.normalize(model.module.prototypes_tar_R), F.normalize(source_NI_global)) + criterion_mmd(F.normalize(model.module.prototypes_tar_N), F.normalize(source_TI_global)) + criterion_mmd(F.normalize(model.module.prototypes_tar_T), F.normalize(source_RGB_global)) + \
                                criterion_mmd(F.normalize(model.module.prototypes_tar_R), F.normalize(source_TI_global)) + criterion_mmd(F.normalize(model.module.prototypes_tar_N), F.normalize(source_RGB_global)) + criterion_mmd(F.normalize(model.module.prototypes_tar_T), F.normalize(source_NI_global))
                
                loss_mmd = loss_mmd_1 + loss_mmd_2 + loss_mmd_3 + loss_mmd_4  
        

                #XBM single
                #_________________________________________________________________________________________________________________________
                loss_xbm = 0
                ori_feats = torch.cat([source_ori,target_ori])
                targets = torch.cat([target_source,target_target])
                xbm.enqueue_dequeue(ori_feats.detach(), targets.detach())
                xbm_feats, xbm_targets = xbm.get()               

                loss_xbm = loss_xbm + criterion_tri_xbm(ori_feats, targets, xbm_feats, xbm_targets) 
                losses_xbm.update(loss_xbm.item())
                #_________________________________________________________________________________________________________________________
                
                lambda1 = 1.0
                lambda2 = 0.8 # 0.8
                
                if cfg.TARGET_DATASETS.NAMES == 'MSVwild863_neo' :
                    loss = loss_tri + loss_xbm + lambda1*loss_mmd + lambda2*loss_3m 
                else:
                    loss = loss_ce + loss_tri + loss_xbm + lambda1*loss_mmd + lambda2*loss_3m                  

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
                    
            acc_r_source = (source_RGB_score.max(1)[1] == target_source).float().mean()
            acc_n_source = (source_NI_score.max(1)[1] == target_source).float().mean()
            acc_t_source = (source_TI_score.max(1)[1] == target_source).float().mean()

            acc_r_target = (target_RGB_score.max(1)[1] == target_target).float().mean()
            acc_n_target = (target_NI_score.max(1)[1] == target_target).float().mean()
            acc_t_target = (target_TI_score.max(1)[1] == target_target).float().mean()  

            loss_meter.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            losses_mmd.update(loss_mmd.item())
            losees_3m.update(loss_3m.item())
            acc_r_source_meter.update(acc_r_source, 1)
            acc_n_source_meter.update(acc_n_source, 1)
            acc_t_source_meter.update(acc_t_source, 1)
            acc_r_target_meter.update(acc_r_target, 1)
            acc_n_target_meter.update(acc_n_target, 1)
            acc_t_target_meter.update(acc_t_target, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                # print(scheduler.._get_lr(epoch))
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, loss_ce: {:.3f}, loss_tri: {:.3f}, loss_xbm: {:.3f}, losses_mmd: {:.3f}, losses_mmm: {:.3f}, Acc_s_rgb: {:.3f},  Acc_s_ni: {:.3f}, Acc_s_ti: {:.3f}, Acc_t_rgb: {:.3f}, Acc_t_ni: {:.3f}, Acc_t_ti: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, losses_ce.avg, losses_tri.avg, losses_xbm.avg, losses_mmd.avg, losees_3m.avg, acc_r_source_meter.avg, acc_n_source_meter.avg, acc_t_source_meter.avg, acc_r_target_meter.avg, acc_n_target_meter.avg, acc_t_target_meter.avg, scheduler._get_lr(epoch)[0]))
        scheduler.step(epoch)
        # model.module.set_prototype_update_weight(epoch, epochs, cfg)
        
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
       
        logger.info("Epoch {} done. total time: {:.3f}[s] Time per batch: {:.3f}[s] Speed: {:.2f}[samples/s]"
                    .format(epoch, end_time - start_time, time_per_batch, cfg.SOLVER.IMS_PER_BATCH / time_per_batch))

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                            os.path.join(output_dir, cfg.DATASETS.NAMES + '_2_' + cfg.TARGET_DATASETS.NAMES + '_{0}_{1}.pth'.format(epoch,cfg.SOLVER.SEED)))

        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = {'RGB': img['RGB'].to(device),
                            'NI': img['NI'].to(device),
                            'TI': img['TI'].to(device)}
                    
                    scenceids = target_view
                    if cfg.MODEL.SIE_CAMERA:
                        camids = camids.to(device)
                    else: 
                        camids = None
                    if cfg.MODEL.SIE_VIEW:
                        target_view = target_view.to(device)
                    else: 
                        target_view = None
                    
                    feat,feat1,feat2,feat3 = model(img, cam_label=camids, view_label=target_view, domain=0)
                    
                    if cfg.DATASETS.NAMES in ["MSVR310", "MSVR310_neo","MSVR310_neo_100"]:
                        evaluator1.update((feat, vid, camid, scenceids, _))
                    elif cfg.DATASETS.NAMES == "alldayRNTG" :
                        evaluator1.update((feat, vid, camid, scenceids))
                    else:
                        evaluator1.update((feat, vid, camid, _))
            start_time = time.time()
            for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(target_val_loader):
                with torch.no_grad():
                    img = {'RGB': img['RGB'].to(device),
                            'NI': img['NI'].to(device),
                            'TI': img['TI'].to(device)}
                    
                    scenceids = target_view
                    if cfg.MODEL.SIE_CAMERA:
                        camids = camids.to(device)
                    else: 
                        camids = None
                    if cfg.MODEL.SIE_VIEW:
                        target_view = target_view.to(device)
                    else: 
                        target_view = None
                    
                    feat,feat1,feat2,feat3 = model(img, cam_label=camids, view_label=target_view)
                        
                    if cfg.TARGET_DATASETS.NAMES in ["MSVR310", "MSVR310_neo","MSVR310_neo_100"]:
                        evaluator.update((feat, vid, camid, scenceids, _))
                    elif cfg.TARGET_DATASETS.NAMES == "alldayRNTG" :
                        evaluator.update((feat, vid, camid, scenceids))
                    else:
                        evaluator.update((feat, vid, camid, _))
            end_time = time.time()
            logger.info("Inference total time: {:.3f}[s]"
                        .format(end_time - start_time))
            cmc, mAP, _, _, _, _, _ = evaluator1.compute()
            logger.info("validating on dataset:{}".format(cfg.DATASETS.NAMES))
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.2%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
            if mAP >= best_index1['mAP']:
                best_index1['mAP'] = mAP
                best_index1['Rank-1'] = cmc[0]
                best_index1['Rank-5'] = cmc[4]
                best_index1['Rank-10'] = cmc[9]
                torch.save(model.state_dict(),
                            os.path.join(output_dir, cfg.DATASETS.NAMES + '_{}_best.pth'.format(cfg.SOLVER.SEED)))
            logger.info("Best mAP: {:.2%}".format(best_index1['mAP']))
            logger.info("Best Rank-1: {:.2%}".format(best_index1['Rank-1']))
            logger.info("Best Rank-5: {:.2%}".format(best_index1['Rank-5']))
            logger.info("Best Rank-10: {:.2%}".format(best_index1['Rank-10']))
                                                            
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("validating on dataset:{}".format(cfg.TARGET_DATASETS.NAMES))
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.2%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
            if mAP >= best_index['mAP']:
                best_index['mAP'] = mAP
                best_index['Rank-1'] = cmc[0]
                best_index['Rank-5'] = cmc[4]
                best_index['Rank-10'] = cmc[9]
                torch.save(model.state_dict(),
                            os.path.join(output_dir, cfg.TARGET_DATASETS.NAMES + '_{}_best.pth'.format(cfg.SOLVER.SEED)))
            logger.info("Best mAP: {:.2%}".format(best_index['mAP']))
            logger.info("Best Rank-1: {:.2%}".format(best_index['Rank-1']))
            logger.info("Best Rank-5: {:.2%}".format(best_index['Rank-5']))
            logger.info("Best Rank-10: {:.2%}".format(best_index['Rank-10']))
            torch.cuda.empty_cache()
