python train_comet_ssd.py --dataset "VOC300" \
                          --dataset_root $1 \
                          --batch_size 16 \
                          --num_workers 4 \
                          --lr 1e-3 \
                          --save_iter 10000 \
                          --cuda True \
                          --model_name comet_csd300_VOC300
                          # --save_epoch 999999999999 \
                          # --warmup_period 20000 \
                          # --visdom True
