#### NON-ROBUST DATASET ####
python3 evaluate_model.py \
  --dataset cifar10 \
  --arch resnet50_bn_cifar \
  --evaluation-dataset cifar10 \
  --model-path /workspace/Projects/research-collections/code/models/fnb/20210914_082821_595_d_non_robust_cifar_resnet50_bn_cifar_1337 \
  --batch-size 128

PGD_EPSILON="0.25 0.5"
PGD_ITERATIONS=20
for eps in $PGD_EPSILON; do
  python3 evaluate_model.py \
          --dataset cifar10 \
          --arch resnet50_bn_cifar \
          --evaluation-dataset cifar10 \
          --model-path /workspace/Projects/research-collections/code/models/fnb/20210914_082821_595_d_non_robust_cifar_resnet50_bn_cifar_1337 \
          --batch-size 128 \
          --attack-name pgd_l2 \
          --attack-criterion NegXent \
          --pgd-epsilon $eps \
          --pgd-iterations $PGD_ITERATIONS \
          --pgd-step-size 2.5*$eps/$PGD_ITERATIONS \
          --postfix _$eps
done

#### ROBUST DATASET ####
python3 evaluate_model.py \
  --dataset cifar10 \
  --arch resnet50_bn_cifar \
  --evaluation-dataset cifar10 \
  --model-path /workspace/Projects/research-collections/code/models/fnb/20210914_083231_297_d_robust_cifar_resnet50_bn_cifar_1337 \
  --batch-size 128

PGD_EPSILON="0.25 0.5"
PGD_ITERATIONS=20
for eps in $PGD_EPSILON; do
  python3 evaluate_model.py \
          --dataset cifar10 \
          --arch resnet50_bn_cifar \
          --evaluation-dataset cifar10 \
          --model-path /workspace/Projects/research-collections/code/models/fnb/20210914_083231_297_d_robust_cifar_resnet50_bn_cifar_1337 \
          --batch-size 128 \
          --attack-name pgd_l2 \
          --attack-criterion NegXent \
          --pgd-epsilon $eps \
          --pgd-iterations $PGD_ITERATIONS \
          --pgd-step-size 2.5*$eps/$PGD_ITERATIONS \
          --postfix _$eps
done

#### DDET DATASET ####
python3 evaluate_model.py \
  --dataset cifar10 \
  --arch resnet50_bn_cifar \
  --evaluation-dataset cifar10 \
  --model-path /workspace/Projects/research-collections/code/models/fnb/20210914_083357_574_ddet_cifar_resnet50_bn_cifar_1337 \
  --batch-size 128

PGD_EPSILON="0.25 0.5"
PGD_ITERATIONS=20
for eps in $PGD_EPSILON; do
  python3 evaluate_model.py \
          --dataset cifar10 \
          --arch resnet50_bn_cifar \
          --evaluation-dataset cifar10 \
          --model-path /workspace/Projects/research-collections/code/models/fnb/20210914_083357_574_ddet_cifar_resnet50_bn_cifar_1337 \
          --batch-size 128 \
          --attack-name pgd_l2 \
          --attack-criterion NegXent \
          --pgd-epsilon $eps \
          --pgd-iterations $PGD_ITERATIONS \
          --pgd-step-size 2.5*$eps/$PGD_ITERATIONS \
          --postfix _$eps
done

### DRAND DATASET ####
python3 evaluate_model.py \
  --dataset cifar10 \
  --arch resnet50_bn_cifar \
  --evaluation-dataset cifar10 \
  --model-path /workspace/Projects/research-collections/code/models/fnb/20210914_083436_753_drand_cifar_resnet50_bn_cifar_1337 \
  --batch-size 128

PGD_EPSILON="0.25 0.5"
PGD_ITERATIONS=20
for eps in $PGD_EPSILON; do
  CUDA_VISIBLE_DEVICES=3 python3 evaluate_model.py \
          --dataset cifar10 \
          --arch resnet50_bn_cifar \
          --evaluation-dataset cifar10 \
          --model-path /workspace/Projects/research-collections/code/models/fnb/20210914_083436_753_drand_cifar_resnet50_bn_cifar_1337 \
          --batch-size 128 \
          --attack-name pgd_l2 \
          --attack-criterion NegXent \
          --pgd-epsilon $eps \
          --pgd-iterations $PGD_ITERATIONS \
          --pgd-step-size 2.5*$eps/$PGD_ITERATIONS \
          --postfix _$eps
done