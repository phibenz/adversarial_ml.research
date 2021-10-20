#### NON-ROBUST DATASET ####
# python3 evaluate_model.py \
#   --dataset cifar10 \
#   --arch resnet50_bn_cifar \
#   --evaluation-dataset cifar10 \
#   --model-path /workspace/Projects/research-collections/code/models/grad_imgs/20211020_061601_528_grad_imgs_cifar10_resnet50_bn_cifar_1337_non_robust_cifar \
#   --batch-size 128

PGD_EPSILON="0.25 0.5"
PGD_ITERATIONS=20
for eps in $PGD_EPSILON; do
  python3 evaluate_model.py \
          --dataset cifar10 \
          --arch resnet50_bn_cifar \
          --evaluation-dataset cifar10 \
          --model-path /workspace/Projects/research-collections/code/models/grad_imgs/20211020_061601_528_grad_imgs_cifar10_resnet50_bn_cifar_1337_non_robust_cifar \
          --batch-size 128 \
          --attack-name pgd_l2 \
          --attack-criterion NegXent \
          --pgd-epsilon $eps \
          --pgd-iterations $PGD_ITERATIONS \
          --pgd-step-size 2.5*$eps/$PGD_ITERATIONS \
          --postfix _$eps
done

#### ROBUST DATASET ####
# python3 evaluate_model.py \
#   --dataset cifar10 \
#   --arch resnet50_bn_cifar \
#   --evaluation-dataset cifar10 \
#   --model-path /workspace/Projects/research-collections/code/models/grad_imgs/20211020_061828_085_grad_imgs_cifar10_resnet50_bn_cifar_1337_robust_cifar \
#   --batch-size 128

PGD_EPSILON="0.25 0.5"
PGD_ITERATIONS=20
for eps in $PGD_EPSILON; do
  python3 evaluate_model.py \
          --dataset cifar10 \
          --arch resnet50_bn_cifar \
          --evaluation-dataset cifar10 \
          --model-path /workspace/Projects/research-collections/code/models/grad_imgs/20211020_061828_085_grad_imgs_cifar10_resnet50_bn_cifar_1337_robust_cifar \
          --batch-size 128 \
          --attack-name pgd_l2 \
          --attack-criterion NegXent \
          --pgd-epsilon $eps \
          --pgd-iterations $PGD_ITERATIONS \
          --pgd-step-size 2.5*$eps/$PGD_ITERATIONS \
          --postfix _$eps
done

#### DDET DATASET ####
# python3 evaluate_model.py \
#   --dataset cifar10 \
#   --arch resnet50_bn_cifar \
#   --evaluation-dataset cifar10 \
#   --model-path /workspace/Projects/research-collections/code/models/grad_imgs/20211019_052252_006_grad_imgs_cifar10_resnet50_bn_cifar_1337_ddet \
#   --batch-size 128

# PGD_EPSILON="0.25 0.5"
# PGD_ITERATIONS=20
# for eps in $PGD_EPSILON; do
#   python3 evaluate_model.py \
#           --dataset cifar10 \
#           --arch resnet50_bn_cifar \
#           --evaluation-dataset cifar10 \
#           --model-path /workspace/Projects/research-collections/code/models/grad_imgs/20211019_052252_006_grad_imgs_cifar10_resnet50_bn_cifar_1337_ddet \
#           --batch-size 128 \
#           --attack-name pgd_l2 \
#           --attack-criterion NegXent \
#           --pgd-epsilon $eps \
#           --pgd-iterations $PGD_ITERATIONS \
#           --pgd-step-size 2.5*$eps/$PGD_ITERATIONS \
#           --postfix _$eps
# done

### DRAND DATASET ####
# python3 evaluate_model.py \
#   --dataset cifar10 \
#   --arch resnet50_bn_cifar \
#   --evaluation-dataset cifar10 \
#   --model-path /workspace/Projects/research-collections/code/models/grad_imgs/20211019_052550_083_grad_imgs_cifar10_resnet50_bn_cifar_1337_drand \
#   --batch-size 128

# PGD_EPSILON="0.25 0.5"
# PGD_ITERATIONS=20
# for eps in $PGD_EPSILON; do
#   python3 evaluate_model.py \
#           --dataset cifar10 \
#           --arch resnet50_bn_cifar \
#           --evaluation-dataset cifar10 \
#           --model-path /workspace/Projects/research-collections/code/models/grad_imgs/20211019_052550_083_grad_imgs_cifar10_resnet50_bn_cifar_1337_drand \
#           --batch-size 128 \
#           --attack-name pgd_l2 \
#           --attack-criterion NegXent \
#           --pgd-epsilon $eps \
#           --pgd-iterations $PGD_ITERATIONS \
#           --pgd-step-size 2.5*$eps/$PGD_ITERATIONS \
#           --postfix _$eps
# done