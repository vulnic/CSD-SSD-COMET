python train_ssd.py --dataset "VOC300" \
                    --dataset_root $1 \
                    --batch_size 32 \
                    --num_workers 4 \
                    --lr 1.5e-3 \
                    --save_iter 10000 \
                    --model_name base_ssd300_VOC07
                    # --warmup_period 20000 \
                    # --save_epoch 999999999999 \
                    # --visdom True 
                    # learning rate lowered -> 1e-3*64/32 = 2e-3 -> 1.5e-3
