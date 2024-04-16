# CUDA_VISIBLE_DEVICES=0 bash mvcnn/train/train_mvcnn.sh
# CUDA_VISIBLE_DEVICES=3 nohup bash mvcnn/train/train_mvcnn.sh >> mvcnn/output/logs/train_mvcnn_50.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -m mvcnn.train.train_mvcnn \
    --name "MVCNN_"\
    --batchSize 8\
    --num_models 1000\
    --lr 5e-5\
    --weight_decay 0.0\
    --cnn_name "vgg11"\
    --num_views 12\
    --img_log_dir "mvcnn/output/logs"\
    --img_log_name "Model Performance "\
    --device "cuda:0"\
    --save_dir "mvcnn/output/model_50"\
    --epochs 30\
    --eval_step 200\
    --logging_step 50\
    --train_path "data/modelnet40_images_new_12x/*/train"\
    --val_path "data/modelnet40_images_new_12x/*/test"\
    --noise_ratio 0.5\
    --image_load_path_train 'data/image/train_image.npz'\
    --image_save_path_train 'data/image/train_image.npz'\
    --image_load_path_test 'data/image/test_image.npz'\
    --image_save_path_test 'data/image/test_image.npz'\
    --image_load_path_test_multi 'data/image/test_image_enhanced.npz'