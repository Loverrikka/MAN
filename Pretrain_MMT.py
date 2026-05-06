import os
from config import cfg
# os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
from utils.logger import setup_logger
from data import make_dataloader,make_dataloader_DHCCN
from modeling import make_model
from solver.make_optimizer import make_optimizer,make_optimizer_clip
from solver.scheduler_factory import create_scheduler
import random
import torch
import numpy as np
import argparse
import time
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP, R1_mAP_eval_allday
from torch.cuda import amp
from layers import CrossEntropyLabelSmooth, TripletLoss,SoftTripletLoss
from solver.lr_scheduler import WarmupMultiStepLR


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Multimodal_DA_ReID PreTraining")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "--model_sl", default=None, help="select_model", type=int
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
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)
    
    output_dir = cfg.OUTPUT_DIR
    if cfg.MODEL.BASE == 9 :
        out = 0
    else:
        print('model chioce error !')

        
    output_dir = output_dir[out]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("Multimodal_DA_ReID", output_dir, cfg, if_train=True)
    logger.info("Saving model in the path :{}".format(output_dir))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    dataset, target_dataset, train_loader, source_train_loader, target_train_loader, val_loader, target_val_loader, num_query, target_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    print("data is ready")
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.cuda()
    
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
    
    criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
    # criterion_triple = SoftTripletLoss(margin=0.0).cuda()
    criterion_triple = TripletLoss(margin=cfg.SOLVER.MARGIN).cuda()

    loss_meter = AverageMeter()
    losses_ce = AverageMeter()
    losses_tri = AverageMeter()
    acc_meter = AverageMeter()
    
    if cfg.DATASETS.NAMES  in ["MSVR310", "MSVR310_neo","MSVR310_neo_100"]:
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    elif cfg.DATASETS.NAMES == "alldayRNTG" :
        evaluator = R1_mAP_eval_allday(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    
    scaler = amp.GradScaler()
    # train
    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}

    #--------------------------------------------------------------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        losses_ce.reset()
        losses_tri.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
            optimizer.zero_grad()
            img = {'RGB': img['RGB'].to(device),
                'NI': img['NI'].to(device),
                'TI': img['TI'].to(device)}
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                
                ori_score, ori,RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global = model(img)
                loss = torch.tensor(0.0).cuda()
                loss_ce = criterion_ce(ori_score, target) + criterion_ce(RGB_score, target) + criterion_ce(NI_score, target) + criterion_ce(TI_score, target)
                loss_tri = criterion_triple(ori,target) + criterion_triple(RGB_global,target) + criterion_triple(NI_global,target) + criterion_triple(TI_global,target)
                loss = loss_ce + loss_tri

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            acc = (ori_score.max(1)[1] == target).float().mean()


            loss_meter.update(loss.item(), img['RGB'].shape[0])
            losses_ce.update(loss_ce.item(),img['RGB'].shape[0])
            losses_tri.update(loss_tri.item(),img['RGB'].shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                # print(scheduler._get_lr(epoch))
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, loss_ce: {:.3f}, loss_tri: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, losses_ce.avg, losses_tri.avg,  acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
    
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.2f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                        os.path.join(output_dir, cfg.DATASETS.NAMES + '_{0}_{1}.pth'.format(epoch,cfg.SOLVER.SEED)))

        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = {'RGB': img['RGB'].to(device),
                            'NI': img['NI'].to(device),
                            'TI': img['TI'].to(device)}
                    camids = camids.to(device)
                    scenceids = target_view
                    target_view = target_view.to(device)
                    feat,feat1,feat2,feat3 = model(img, cam_label=camids, view_label=target_view)
                    if cfg.DATASETS.NAMES in ["MSVR310", "MSVR310_neo","MSVR310_neo_100"]:
                        evaluator.update((feat, vid, camid, scenceids, _))
                    elif cfg.DATASETS.NAMES == "alldayRNTG" :
                        evaluator.update((feat, vid, camid, scenceids))
                    else:
                        evaluator.update((feat, vid, camid))
                        
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
                            os.path.join(output_dir, cfg.DATASETS.NAMES + '_{}_best.pth'.format(cfg.SOLVER.SEED)))
            logger.info("Best mAP: {:.2%}".format(best_index['mAP']))
            logger.info("Best Rank-1: {:.2%}".format(best_index['Rank-1']))
            logger.info("Best Rank-5: {:.2%}".format(best_index['Rank-5']))
            logger.info("Best Rank-10: {:.2%}".format(best_index['Rank-10']))
            torch.cuda.empty_cache()


