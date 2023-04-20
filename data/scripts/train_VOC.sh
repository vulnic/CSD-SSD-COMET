cd "/mnt/c/Users/vlade813/Desktop/School/Masters/Spring 2023/Deep Learning/CSD-SSD-COMET"
python train_ssd.py --dataset "VOC0712_300" \
                    --dataset_root "/mnt/c/Users/vlade813/Desktop/School/Masters/Spring 2023/Deep Learning/Final Project/" \
                    --batch_size 50 \
                    --num_workers 4 \
                    --lr 1.5e-3 \
                    --warmup_period 20000 \
                    --save_iter 10000 \
                    --save_epoch 999999999999 \
                    --model_name base_ssd300_VOC0712
                    # --visdom True 
                    # learning rate lowered -> 1e-3*64/32 = 2e-3 -> 1.5e-3