# CUDA_VISIBLE_DEVICES=0 bash mvcnn_elr/train/train.sh
# CUDA_VISIBLE_DEVICES=5 nohup bash mvcnn_elr/train/train.sh > mvcnn_elr/output/eval/logs/train_00.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python -m mvcnn_elr.train.train \
    --name "MVCNN_ELR_"\
    --batchSize 8\
    --num_models 1000\
    --lr 5e-5\
    --weight_decay 0.0\
    --cnn_name "vgg11"\
    --num_views 12\
    --img_log_dir "mvcnn_elr/output/eval/logs"\
    --img_log_name "Model Performance "\
    --monitor "max val_my_metric"\
    --beta 0.7\
    --lambda_ 3.0\
    --device "cuda:0"\
    --save_dir "mvcnn_elr/output/eval/model_00"\
    --epochs 30\
    --eval_step 200\
    --logging_step 50\
    --early_stop 10000000\
    --train_path "data/modelnet40_images_new_12x/*/train"\
    --val_path "data/modelnet40_images_new_12x/*/test"\
    --noise_path None\
    --noise_ratio 0.0 \
    --image_load_path_train 'data/image/train_image.npz'\
    --image_save_path_train 'data/image/train_image.npz'\
    --image_load_path_test 'data/image/test_image.npz'\
    --image_save_path_test 'data/image/test_image.npz'\
    --image_load_path_test_multi 'data/image/test_image_enhanced.npz'