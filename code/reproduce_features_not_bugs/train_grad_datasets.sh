#### NON-ROBUST DATASET ####
python3 train_model.py \
  --dataset grad_imgs_cifar10 \
  --grad-imgs-path /workspace/Projects/adversarial_ml.research/code/models/fnb/20210906_075452_407_cifar10_resnet50_bn_cifar_1337_nat/grad_datasets/20211019_131444_944_non_robust_cifar \
  --arch resnet50_bn_cifar \
  --epochs 150 \
  --batch-size 128 \
  --learning-rate 0.01 \
  --weight-decay 5e-4 \
  --schedule 50 100 \
  --gamma 0.1 \
  --subfolder grad_imgs \
  --postfix _non_robust_cifar

#### ROBUST DATASET ####
python3 train_model.py \
  --dataset grad_imgs_cifar10 \
  --grad-imgs-path /workspace/Projects/adversarial_ml.research/code/models/fnb/20210906_075509_644_cifar10_resnet50_bn_cifar_1337_adv_train_l2_0_5/grad_datasets/20211018_144037_805_robust_cifar \
  --arch resnet50_bn_cifar \
  --epochs 150 \
  --batch-size 128 \
  --learning-rate 0.1 \
  --weight-decay 5e-4 \
  --schedule 50 100 \
  --gamma 0.1 \
  --subfolder grad_imgs \
  --postfix _robust_cifar

#### DDET DATASET ####
python3 train_model.py \
  --dataset grad_imgs_cifar10 \
  --grad-imgs-path /workspace/Projects/adversarial_ml.research/code/models/fnb/20210906_075452_407_cifar10_resnet50_bn_cifar_1337_nat/grad_datasets/20211018_144116_447_ddet_cifar \
  --arch resnet50_bn_cifar \
  --epochs 150 \
  --batch-size 128 \
  --learning-rate 0.1 \
  --weight-decay 5e-4 \
  --schedule 50 100 \
  --gamma 0.1 \
  --augmentation False \
  --subfolder grad_imgs

### DRAND DATASET ####
python3 train_model.py \
  --dataset grad_imgs_cifar10 \
  --grad-imgs-path /workspace/Projects/adversarial_ml.research/code/models/fnb/20210906_075452_407_cifar10_resnet50_bn_cifar_1337_nat/grad_datasets/20211018_222720_372_drand_cifar \
  --arch resnet50_bn_cifar \
  --epochs 150 \
  --batch-size 128 \
  --learning-rate 0.01 \
  --weight-decay 5e-4 \
  --schedule 50 100 \
  --gamma 0.1 \
  --subfolder grad_imgs
