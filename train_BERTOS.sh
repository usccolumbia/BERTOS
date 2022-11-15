python train_BERTOS.py  \
       --config_name ./random_config/  \
       --dataset_name materials_icsd.py   \
       --max_length 100  \
       --per_device_train_batch_size 256 \
       --learning_rate 1e-3  \
       --num_train_epochs 500    \
       --output_dir ./icsd