### CIFAR10 ResNet50 ###
python3 evaluate_model.py \
        --dataset cifar10 \
        --arch resnet50_bn_cifar \
        --evaluation-dataset cifar10 \
        --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075452_407_cifar10_resnet50_bn_cifar_1337_nat \
        --batch-size 128

PGD_EPSILON="0.25 0.5 1.0 2.0"
PGD_ITERATIONS=20
for eps in $PGD_EPSILON; do
  python3 evaluate_model.py \
          --dataset cifar10 \
          --arch resnet50_bn_cifar \
          --evaluation-dataset cifar10 \
          --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075452_407_cifar10_resnet50_bn_cifar_1337_nat \
          --batch-size 128 \
          --attack-name pgd_l2 \
          --attack-criterion NegXent \
          --pgd-epsilon $eps \
          --pgd-iterations $PGD_ITERATIONS \
          --pgd-step-size 2.5*$eps/$PGD_ITERATIONS \
          --postfix _$eps
done

PGD_EPSILON="8/255 16/255"
PGD_ITERATIONS=20
for eps in $PGD_EPSILON; do
  eps_stripped=$(echo $eps | cut -d '/' -f 1 )
  python3 evaluate_model.py \
        --dataset cifar10 \
        --arch resnet50_bn_cifar \
        --evaluation-dataset cifar10 \
        --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075452_407_cifar10_resnet50_bn_cifar_1337_nat \
        --batch-size 128 \
        --attack-name pgd_linf \
        --attack-criterion NegXent \
        --pgd-epsilon $eps \
        --pgd-iterations $PGD_ITERATIONS \
        --pgd-step-size 2.5*$eps/$PGD_ITERATIONS \
        --postfix _${eps_stripped}
done

### CIFAR10 ResNet50 L2 0.25 ###
python3 evaluate_model.py \
        --dataset cifar10 \
        --arch resnet50_bn_cifar \
        --evaluation-dataset cifar10 \
        --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075501_166_cifar10_resnet50_bn_cifar_1337_adv_train_l2_0_25 \
        --batch-size 128

PGD_EPSILON="0.25 0.5 1.0 2.0"
PGD_ITERATIONS=20
for eps in $PGD_EPSILON; do
  python3 evaluate_model.py \
          --dataset cifar10 \
          --arch resnet50_bn_cifar \
          --evaluation-dataset cifar10 \
          --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075501_166_cifar10_resnet50_bn_cifar_1337_adv_train_l2_0_25 \
          --batch-size 128 \
          --attack-name pgd_l2 \
          --attack-criterion NegXent \
          --pgd-epsilon $eps \
          --pgd-iterations $PGD_ITERATIONS \
          --pgd-step-size 2.5*$eps/$PGD_ITERATIONS \
          --postfix _$eps
done

### CIFAR10 ResNet50 L2 0.5 ###
python3 evaluate_model.py \
        --dataset cifar10 \
        --arch resnet50_bn_cifar \
        --evaluation-dataset cifar10 \
        --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075509_644_cifar10_resnet50_bn_cifar_1337_adv_train_l2_0_5 \
        --batch-size 128

PGD_EPSILON="0.25 0.5 1.0 2.0"
PGD_ITERATIONS=20
for eps in $PGD_EPSILON; do
  python3 evaluate_model.py \
          --dataset cifar10 \
          --arch resnet50_bn_cifar \
          --evaluation-dataset cifar10 \
          --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075509_644_cifar10_resnet50_bn_cifar_1337_adv_train_l2_0_5 \
          --batch-size 128 \
          --attack-name pgd_l2 \
          --attack-criterion NegXent \
          --pgd-epsilon $eps \
          --pgd-iterations $PGD_ITERATIONS \
          --pgd-step-size 2.5*$eps/$PGD_ITERATIONS \
          --postfix _$eps
done

#### CIFAR10 ResNet50 L2 1.0 ####
python3 evaluate_model.py \
        --dataset cifar10 \
        --arch resnet50_bn_cifar \
        --evaluation-dataset cifar10 \
        --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075518_497_cifar10_resnet50_bn_cifar_1337_adv_train_l2_1_0 \
        --batch-size 128

PGD_EPSILON="0.25 0.5 1.0 2.0"
PGD_ITERATIONS=20
for eps in $PGD_EPSILON; do
  python3 evaluate_model.py \
          --dataset cifar10 \
          --arch resnet50_bn_cifar \
          --evaluation-dataset cifar10 \
          --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075518_497_cifar10_resnet50_bn_cifar_1337_adv_train_l2_1_0 \
          --batch-size 128 \
          --attack-name pgd_l2 \
          --attack-criterion NegXent \
          --pgd-epsilon $eps \
          --pgd-iterations $PGD_ITERATIONS \
          --pgd-step-size 2.5*$eps/$PGD_ITERATIONS \
          --postfix _$eps
done

### CIFAR10 ResNet50 Linf 8/255 ###
python3 evaluate_model.py \
        --dataset cifar10 \
        --arch resnet50_bn_cifar \
        --evaluation-dataset cifar10 \
        --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075529_035_cifar10_resnet50_bn_cifar_1337_adv_train_linf_8 \
        --batch-size 128

PGD_EPSILON="8/255 16/255"
PGD_ITERATIONS=20
for eps in $PGD_EPSILON; do
  eps_stripped=$(echo $eps | cut -d '/' -f 1 )
  python3 evaluate_model.py \
          --dataset cifar10 \
          --arch resnet50_bn_cifar \
          --evaluation-dataset cifar10 \
          --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075529_035_cifar10_resnet50_bn_cifar_1337_adv_train_linf_8 \
          --batch-size 128 \
          --attack-name pgd_linf \
          --attack-criterion NegXent \
          --pgd-epsilon $eps \
          --pgd-iterations $PGD_ITERATIONS \
          --pgd-step-size 2.5*$eps/$PGD_ITERATIONS \
          --postfix _$eps_stripped
done

### ImageNet ResNet50 ###
python3 evaluate_model.py \
        --dataset imagenet \
        --arch resnet50 \
        --evaluation-dataset imagenet \
        --batch-size 32

PGD_EPSILON="0.5 1.0 2.0 3.0"
PGD_ITERATIONS=20
for eps in $PGD_EPSILON; do
  python3 evaluate_model.py \
          --dataset imagenet \
          --arch resnet50 \
          --evaluation-dataset imagenet \
          --batch-size 32 \
          --attack-name pgd_l2 \
          --attack-criterion NegXent \
          --pgd-epsilon $eps \
          --pgd-iterations $PGD_ITERATIONS \
          --pgd-step-size 2.5*$eps/$PGD_ITERATIONS \
          --postfix _$eps
done

PGD_EPSILON="4/255 8/255 16/255"
PGD_ITERATIONS=20
for eps in $PGD_EPSILON; do
  eps_stripped=$(echo $eps | cut -d '/' -f 1 )
  python3 evaluate_model.py \
          --dataset imagenet \
          --arch resnet50 \
          --evaluation-dataset imagenet \
          --batch-size 32 \
          --attack-name pgd_linf \
          --attack-criterion NegXent \
          --pgd-epsilon $eps \
          --pgd-iterations $PGD_ITERATIONS \
          --pgd-step-size 2.5*$eps/$PGD_ITERATIONS \
          --postfix _$eps_stripped
done

### ImageNet ResNet50 L2 3.0 ###
python3 evaluate_model.py \
        --dataset imagenet \
        --arch resnet50 \
        --evaluation-dataset imagenet \
        --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075536_549_imagenet_resnet50_1337_adv_train_l2_3_0 \
        --batch-size 32

PGD_EPSILON="0.5 1.0 2.0 3.0"
PGD_ITERATIONS=20
for eps in $PGD_EPSILON; do
  CUDA_VISIBLE_DEVICES=1 python3 evaluate_model.py \
          --dataset imagenet \
          --arch resnet50 \
          --evaluation-dataset imagenet \
          --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075536_549_imagenet_resnet50_1337_adv_train_l2_3_0 \
          --batch-size 32 \
          --attack-name pgd_l2 \
          --attack-criterion NegXent \
          --pgd-epsilon $eps \
          --pgd-iterations $PGD_ITERATIONS \
          --pgd-step-size 2.5*$eps/$PGD_ITERATIONS \
          --postfix _$eps
done


### ImageNet ResNet50 Linf 4/288 ####
python3 evaluate_model.py \
        --dataset imagenet \
        --arch resnet50 \
        --evaluation-dataset imagenet \
        --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075545_753_imagenet_resnet50_1337_adv_train_linf_4 \
        --batch-size 32


PGD_EPSILON="4/255 8/255 16/255"
PGD_ITERATIONS=20
for eps in $PGD_EPSILON; do
  eps_stripped=$(echo $eps | cut -d '/' -f 1 )
  CUDA_VISIBLE_DEVICES=3 python3 evaluate_model.py \
          --dataset imagenet \
          --arch resnet50 \
          --evaluation-dataset imagenet \
          --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075545_753_imagenet_resnet50_1337_adv_train_linf_4 \
          --batch-size 32 \
          --attack-name pgd_linf \
          --attack-criterion NegXent \
          --pgd-epsilon $eps \
          --pgd-iterations $PGD_ITERATIONS \
          --pgd-step-size 2.5*$eps/$PGD_ITERATIONS \
          --postfix _$eps_stripped
done

### ImageNet ResNet50 Linf 8/288 ####
python3 evaluate_model.py \
        --dataset imagenet \
        --arch resnet50 \
        --evaluation-dataset imagenet \
        --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075558_187_imagenet_resnet50_1337_adv_train_linf_8 \
        --batch-size 32


PGD_EPSILON="4/255 8/255 16/255"
PGD_ITERATIONS=20
for eps in $PGD_EPSILON; do
  eps_stripped=$(echo $eps | cut -d '/' -f 1 )
  CUDA_VISIBLE_DEVICES=4 python3 evaluate_model.py \
          --dataset imagenet \
          --arch resnet50 \
          --evaluation-dataset imagenet \
          --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075558_187_imagenet_resnet50_1337_adv_train_linf_8 \
          --batch-size 32 \
          --attack-name pgd_linf \
          --attack-criterion NegXent \
          --pgd-epsilon $eps \
          --pgd-iterations $PGD_ITERATIONS \
          --pgd-step-size 2.5*$eps/$PGD_ITERATIONS \
          --postfix _$eps_stripped
done