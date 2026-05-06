
### Reproduction

#### Datasets

##### WMVEID863_neo link: 链接: https://pan.baidu.com/s/14WICt3XC-ceWjSGOa8nX8g?pwd=3p13 提取码: 3p13 
##### MSVR310_neo link: 链接: https://pan.baidu.com/s/1kT8-XjJhgKr1hcQCkzSI4w?pwd=vusx 提取码: vusx 


### train_example.sh

#### Pretrain MODEl

```bash
CUDA_VISIBLE_DEVICES=4 python Pretrain_MMT.py --config_file /data/MAN/configs/MSVwild863_neo/TOP-ReID.yml --model_sl 9
CUDA_VISIBLE_DEVICES=4 python Pretrain_MMT.py --config_file /data/MAN/configs/RGBNT100/TOP-ReID.yml --model_sl 9
CUDA_VISIBLE_DEVICES=4 python Pretrain_MMT.py --config_file /data/MAN/configs/MSVR310_neo/TOP-ReID.yml --model_sl 9
CUDA_VISIBLE_DEVICES=4 python Pretrain_MMT.py --config_file /data/MAN/configs/RGBNT201/TOP-ReID.yml --model_sl 9
CUDA_VISIBLE_DEVICES=4 python Pretrain_MMT.py --config_file /data/MAN/configs/market_to_RGBNT201/TOP-ReID.yml --model_sl 9
```

### train_net_uda_reid_neo 

```bash
CUDA_VISIBLE_DEVICES=3 python train_net_uda_reid_neo.py --config_file /data/MAN/configs/ZZZ_DA_config/RGBNT100_2_MSVwild863_neo/TOP-ReID.yml --model_sl 9 --data_iter True
CUDA_VISIBLE_DEVICES=3 python train_net_uda_reid_neo.py --config_file /data/MAN/configs/ZZZ_DA_config/MSVwild863_neo_2_RGBNT100/TOP-ReID.yml --model_sl 9 --data_iter True
CUDA_VISIBLE_DEVICES=3 python train_net_uda_reid_neo.py --config_file /data/MAN/configs/ZZZ_DA_config/MSVR310_neo_2_MSVwild863_neo/TOP-ReID.yml --model_sl 9 --data_iter True
CUDA_VISIBLE_DEVICES=3 python train_net_uda_reid_neo.py --config_file /data/MAN/configs/ZZZ_DA_config/MSVwild863_neo_2_MSVR310_neo/TOP-ReID.yml --model_sl 9 --data_iter True
CUDA_VISIBLE_DEVICES=3 python train_net_uda_reid_neo.py --config_file /data/MAN/configs/ZZZ_DA_config/RGBNT201_2_market_to_RGBNT201/TOP-ReID.yml --model_sl 9 --data_iter True
CUDA_VISIBLE_DEVICES=3 python train_net_uda_reid_neo.py --config_file /data/MAN/configs/ZZZ_DA_config/market_to_RGBNT201_2_RGBNT201/TOP-ReID.yml --model_sl 9 --data_iter True
```

### train_net_sfda 

```bash
CUDA_VISIBLE_DEVICES=3 python train_net_sfda.py --config_file /data/MAN/configs/ZZZ_DA_config/RGBNT100_2_MSVwild863_neo/TOP-ReID.yml --model_sl 9 --data_iter True
CUDA_VISIBLE_DEVICES=3 python train_net_sfda.py --config_file /data/MAN/configs/ZZZ_DA_config/MSVwild863_neo_2_RGBNT100/TOP-ReID.yml --model_sl 9 --data_iter True
CUDA_VISIBLE_DEVICES=3 python train_net_sfda.py --config_file /data/MAN/configs/ZZZ_DA_config/MSVR310_neo_2_MSVwild863_neo/TOP-ReID.yml --model_sl 9 --data_iter True
CUDA_VISIBLE_DEVICES=3 python train_net_sfda.py --config_file /data/MAN/configs/ZZZ_DA_config/MSVwild863_neo_2_MSVR310_neo/TOP-ReID.yml --model_sl 9 --data_iter True
CUDA_VISIBLE_DEVICES=3 python train_net_sfda.py --config_file /data/MAN/configs/ZZZ_DA_config/RGBNT201_2_market_to_RGBNT201/TOP-ReID.yml --model_sl 9 --data_iter True
CUDA_VISIBLE_DEVICES=3 python train_net_sfda.py --config_file /data/MAN/configs/ZZZ_DA_config/market_to_RGBNT201_2_RGBNT201/TOP-ReID.yml --model_sl 9 --data_iter True
```


## Citation
```bibtex
@article{sheng2025multi,
  title={Multi-level alignment network for unsupervised domain adaptive multi-modality object re-identification},
  author={Sheng, Yusong and Ding, Yuhe and Zheng, Aihua and Liu, Ziqi and Wang, Zi and Tang, Jin},
  journal={Knowledge-Based Systems},
  pages={115015},
  year={2025},
  publisher={Elsevier}
}
```











