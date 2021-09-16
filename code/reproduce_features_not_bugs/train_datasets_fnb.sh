#### NON-ROBUST DATASET ####
python3 train_model.py \
  --dataset d_non_robust_cifar \
  --arch resnet50_bn_cifar \
  --epochs 150 \
  --batch-size 128 \
  --learning-rate 0.01 \
  --weight-decay 5e-4 \
  --schedule 50 100 \
  --gamma 0.1 \
  --subfolder fnb

#### ROBUST DATASET ####
python3 train_model.py \
  --dataset d_robust_cifar \
  --arch resnet50_bn_cifar \
  --epochs 150 \
  --batch-size 128 \
  --learning-rate 0.1 \
  --weight-decay 5e-4 \
  --schedule 50 100 \
  --gamma 0.1 \
  --subfolder fnb

#### DDET DATASET ####
python3 train_model.py \
  --dataset ddet_cifar \
  --arch resnet50_bn_cifar \
  --epochs 150 \
  --batch-size 128 \
  --learning-rate 0.1 \
  --weight-decay 5e-4 \
  --schedule 50 100 \
  --gamma 0.1 \
  --augmentation False \
  --subfolder fnb

### DRAND DATASET ####
python3 train_model.py \
  --dataset drand_cifar \
  --arch resnet50_bn_cifar \
  --epochs 150 \
  --batch-size 128 \
  --learning-rate 0.01 \
  --weight-decay 5e-4 \
  --schedule 50 100 \
  --gamma 0.1 \
  --subfolder fnb
