python train_comet_ssd.py --dataset "VOC300" \
                          --dataset_root "/mnt/c/Users/vlade813/Desktop/School/Masters/Spring_2023/Deep_Learning/VOCdatasets/" \
                          --batch_size 16 \
                          --num_workers 4 \
                          --lr 1e-3 \
                          --save_iter 10000 \
                          --cuda True \
                          --model_name comet_csd300_VOC300
                        #   --save_epoch 999999999999 \
                        #   --warmup_period 20000 \
                          # --visdom True 
                          # learning rate lowered -> 1e-3*64/32 = 2e-3 -> 1.5e-3
