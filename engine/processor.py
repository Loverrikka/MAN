import torch
import torch.nn as nn
from utils.metrics import R1_mAP_eval, R1_mAP, R1_mAP_eval_allday
import pdb

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 logger,
                 modal_sl = None,
                 outnum = 4,
                 ):
    device = "cuda"
    logger.info("Enter inferencing")

    if cfg.TARGET_DATASETS.NAMES in ["MSVR310", "MSVR310_neo","MSVR310_neo_100"]:
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    
    elif cfg.TARGET_DATASETS.NAMES == "alldayRNTG" :
        evaluator = R1_mAP_eval_allday(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,cfg=cfg)
        evaluator.reset()


    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)
            
            if outnum == 4:
                if modal_sl in ['RGB+NIR', 'RGB+TIR', 'NIR+TIR']:
                    feat, feat1, feat2, feat3 = model(img, cam_label=camids, view_label=target_view, modal_sl=modal_sl)
                else:
                    feat, feat1, feat2, feat3 = model(img, cam_label=camids, view_label=target_view)
            elif outnum == 1:
                if modal_sl in ['RGB+NIR', 'RGB+TIR', 'NIR+TIR']:
                    feat = model(img, cam_label=camids, view_label=target_view, modal_sl=modal_sl)
                else:
                    feat = model(img, cam_label=camids, view_label=target_view)
            elif outnum==12:
                if modal_sl in ['RGB+NIR', 'RGB+TIR', 'NIR+TIR']:
                    feat, _, _, _, _, _, _, _, _, _, _, _ = model(img, cam_label=camids, view_label=target_view, modal_sl=modal_sl)
                else:
                    feat, _, _, _, _, _, _, _, _, _, _, _ = model(img, cam_label=camids, view_label=target_view)
            elif outnum==2:
                if modal_sl in ['RGB+NIR', 'RGB+TIR', 'NIR+TIR']:
                    _, feat = model(img, cam_label=camids, view_label=target_view, modal_sl=modal_sl)
                else:
                    _, feat = model(img, cam_label=camids, view_label=target_view)                
            elif outnum==7:
                if modal_sl in ['RGB+NIR', 'RGB+TIR', 'NIR+TIR']:
                    feat, feat1, feat2, feat3, feat4, feat5, feat6 = model(img, cam_label=camids, view_label=target_view, modal_sl=modal_sl)
                else:
                    feat, feat1, feat2, feat3, feat4, feat5, feat6 = model(img, cam_label=camids, view_label=target_view)    
            else:
                raise ValueError(f"Invalid outnum value: {outnum}. Expected 1 or 4.")

            # 根据 modal_sl 设置更新的特征
            if modal_sl == 'RGB':
                feat_to_update = feat1
            elif modal_sl == 'NIR':
                feat_to_update = feat2
            elif modal_sl == 'TIR':
                feat_to_update = feat3
            else:
                feat_to_update = feat

            # 更新 evaluator
            if cfg.TARGET_DATASETS.NAMES in ["MSVR310", "MSVR310_neo", "MSVR310_neo_100"]:
                evaluator.update((feat_to_update, pid, camid, scenceids, imgpath))
            elif cfg.TARGET_DATASETS.NAMES == "alldayRNTG":
                evaluator.update((feat_to_update, pid, camid, scenceids))
            else:
                evaluator.update((feat_to_update, pid, camid, imgpath))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results on {}".format(cfg.TARGET_DATASETS.NAMES))
    logger.info("mAP: {:.2%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


def do_inference_neo(cfg,
                 model,
                 val_loader,
                 num_query,
                 logger,
                 modal_sl = None,
                 outnum=4,
                 ):
    device = "cuda"
    logger.info("Enter inferencing")

    if cfg.DATASETS.NAMES in ["MSVR310", "MSVR310_neo","MSVR310_neo_100"]:
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    
    elif cfg.DATASETS.NAMES == "alldayRNTG" :
        evaluator = R1_mAP_eval_allday(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()


    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)
            
            if outnum == 4:
                if modal_sl in ['RGB+NIR', 'RGB+TIR', 'NIR+TIR']:
                    feat, feat1, feat2, feat3 = model(img, cam_label=camids, view_label=target_view, modal_sl=modal_sl)
                else:
                    feat, feat1, feat2, feat3 = model(img, cam_label=camids, view_label=target_view, domain=0)
            elif outnum == 1:
                if modal_sl in ['RGB+NIR', 'RGB+TIR', 'NIR+TIR']:
                    feat = model(img, cam_label=camids, view_label=target_view, modal_sl=modal_sl)
                else:
                    feat = model(img, cam_label=camids, view_label=target_view)
            elif outnum==12:
                if modal_sl in ['RGB+NIR', 'RGB+TIR', 'NIR+TIR']:
                    feat, _, _, _, _, _, _, _, _, _, _, _ = model(img, cam_label=camids, view_label=target_view, modal_sl=modal_sl)
                else:
                    feat, _, _, _, _, _, _, _, _, _, _, _ = model(img, cam_label=camids, view_label=target_view)
            elif outnum==2:
                if modal_sl in ['RGB+NIR', 'RGB+TIR', 'NIR+TIR']:
                    _, feat = model(img, cam_label=camids, view_label=target_view, modal_sl=modal_sl)
                else:
                    _, feat = model(img, cam_label=camids, view_label=target_view)                
            elif outnum==7:
                if modal_sl in ['RGB+NIR', 'RGB+TIR', 'NIR+TIR']:
                    feat, feat1, feat2, feat3, feat4, feat5, feat6 = model(img, cam_label=camids, view_label=target_view, modal_sl=modal_sl)
                else:
                    feat, feat1, feat2, feat3, feat4, feat5, feat6 = model(img, cam_label=camids, view_label=target_view)  
            else:
                raise ValueError(f"Invalid outnum value: {outnum}. Expected 1 or 4.")

            # 根据 modal_sl 设置更新的特征
            if modal_sl == 'RGB':
                feat_to_update = feat1
            elif modal_sl == 'NIR':
                feat_to_update = feat2
            elif modal_sl == 'TIR':
                feat_to_update = feat3
            else:
                feat_to_update = feat

            # 更新 evaluator
            if cfg.DATASETS.NAMES in ["MSVR310", "MSVR310_neo", "MSVR310_neo_100"]:
                evaluator.update((feat_to_update, pid, camid, scenceids, imgpath))
            elif cfg.DATASETS.NAMES == "alldayRNTG":
                evaluator.update((feat_to_update, pid, camid, scenceids))
            else:
                evaluator.update((feat_to_update, pid, camid, imgpath))

            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results on {}".format(cfg.DATASETS.NAMES))
    logger.info("mAP: {:.2%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
