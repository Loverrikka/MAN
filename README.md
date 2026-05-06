
### Reproduction

#### Datasets

##### WMVEID863_neo link:
##### MSVR310_neo link:







### train_example.sh

#### Pretrain MODEl

##### CUDA_VISIBLE_DEVICES=4 python Pretrain_MMT.py --config_file /data/MAN/configs/MSVwild863_neo/TOP-ReID.yml --model_sl 9
##### CUDA_VISIBLE_DEVICES=4 python Pretrain_MMT.py --config_file /data/MAN/configs/RGBNT100/TOP-ReID.yml --model_sl 9
##### CUDA_VISIBLE_DEVICES=4 python Pretrain_MMT.py --config_file /data/MAN/configs/MSVR310_neo/TOP-ReID.yml --model_sl 9
##### CUDA_VISIBLE_DEVICES=4 python Pretrain_MMT.py --config_file /data/MAN/configs/RGBNT201/TOP-ReID.yml --model_sl 9
##### CUDA_VISIBLE_DEVICES=4 python Pretrain_MMT.py --config_file /data/MAN/configs/market_to_RGBNT201/TOP-ReID.yml --model_sl 9


### train_net_uda_reid_neo 
##### CUDA_VISIBLE_DEVICES=3 python train_net_uda_reid_neo.py --config_file /data/MAN/configs/ZZZ_DA_config/RGBNT100_2_MSVwild863_neo/TOP-ReID.yml --model_sl 9 --data_iter True
##### CUDA_VISIBLE_DEVICES=3 python train_net_uda_reid_neo.py --config_file /data/MAN/configs/ZZZ_DA_config/MSVwild863_neo_2_RGBNT100/TOP-ReID.yml --model_sl 9 --data_iter True
##### CUDA_VISIBLE_DEVICES=3 python train_net_uda_reid_neo.py --config_file /data/MAN/configs/ZZZ_DA_config/MSVR310_neo_2_MSVwild863_neo/TOP-ReID.yml --model_sl 9 --data_iter True
##### CUDA_VISIBLE_DEVICES=3 python train_net_uda_reid_neo.py --config_file /data/MAN/configs/ZZZ_DA_config/MSVwild863_neo_2_MSVR310_neo/TOP-ReID.yml --model_sl 9 --data_iter True
##### CUDA_VISIBLE_DEVICES=3 python train_net_uda_reid_neo.py --config_file /data/MAN/configs/ZZZ_DA_config/RGBNT201_2_market_to_RGBNT201/TOP-ReID.yml --model_sl 9 --data_iter True
##### CUDA_VISIBLE_DEVICES=3 python train_net_uda_reid_neo.py --config_file /data/MAN/configs/ZZZ_DA_config/market_to_RGBNT201_2_RGBNT201/TOP-ReID.yml --model_sl 9 --data_iter True


### train_net_sfda 

##### CUDA_VISIBLE_DEVICES=3 python train_net_sfda.py --config_file /data/MAN/configs/ZZZ_DA_config/RGBNT100_2_MSVwild863_neo/TOP-ReID.yml --model_sl 9 --data_iter True
##### CUDA_VISIBLE_DEVICES=3 python train_net_sfda.py --config_file /data/MAN/configs/ZZZ_DA_config/MSVwild863_neo_2_RGBNT100/TOP-ReID.yml --model_sl 9 --data_iter True
##### CUDA_VISIBLE_DEVICES=3 python train_net_sfda.py --config_file /data/MAN/configs/ZZZ_DA_config/MSVR310_neo_2_MSVwild863_neo/TOP-ReID.yml --model_sl 9 --data_iter True
##### CUDA_VISIBLE_DEVICES=3 python train_net_sfda.py --config_file /data/MAN/configs/ZZZ_DA_config/MSVwild863_neo_2_MSVR310_neo/TOP-ReID.yml --model_sl 9 --data_iter True
##### CUDA_VISIBLE_DEVICES=3 python train_net_sfda.py --config_file /data/MAN/configs/ZZZ_DA_config/RGBNT201_2_market_to_RGBNT201/TOP-ReID.yml --model_sl 9 --data_iter True
##### CUDA_VISIBLE_DEVICES=3 python train_net_sfda.py --config_file /data/MAN/configs/ZZZ_DA_config/market_to_RGBNT201_2_RGBNT201/TOP-ReID.yml --model_sl 9 --data_iter True
















