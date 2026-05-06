from torch import nn
from .. import models
         
def create_vit_model(cfg):
    """
    Create ViT model.
    
    Params:
        cfg: Config instance.
    Returns:
        The TMGF model.
    """
        
    model = models.create(cfg.TMGF.MODEL.ARCH, arch=cfg.TMGF.MODEL.ARCH,
                          img_size=[cfg.TMGF.INPUT.HEIGHT, cfg.TMGF.INPUT.WIDTH], sie_coef=cfg.TMGF.MODEL.SIE_COEF,
                          camera_num=cfg.TMGF.MODEL.SIE_CAMERA, view_num=cfg.TMGF.MODEL.SIE_VIEW,
                          stride_size=cfg.TMGF.MODEL.STRIDE_SIZE, drop_path_rate=cfg.TMGF.MODEL.DROP_PATH,
                          drop_rate=cfg.TMGF.MODEL.DROP_OUT, attn_drop_rate=cfg.TMGF.MODEL.ATTN_DROP_RATE,
                          pretrain_path=cfg.TMGF.MODEL.PRETRAIN_PATH, hw_ratio=cfg.TMGF.MODEL.PRETRAIN_HW_RATIO,
                          gem_pool=cfg.TMGF.MODEL.GEM_POOL, stem_conv=cfg.TMGF.MODEL.STEM_CONV, num_parts=cfg.TMGF.MODEL.NUM_PARTS,
                          has_head=cfg.TMGF.MODEL.HAS_HEAD, global_feature_type=cfg.TMGF.MODEL.GLOBAL_FEATURE_TYPE,
                          granularities=cfg.TMGF.MODEL.GRANULARITIES, branch=cfg.TMGF.MODEL.BRANCH, has_early_feature=cfg.TMGF.MODEL.HAS_EARLY_FEATURE,
                          enable_early_norm=cfg.TMGF.MODEL.ENABLE_EARLY_NORM)
    return model
