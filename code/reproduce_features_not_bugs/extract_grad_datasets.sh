# Reconstruct the d_non_robust_CIFAR dataset
python3 extract_grad_datasets.py \
    --dataset cifar10 \
    --arch resnet50_bn_cifar \
    --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075452_407_cifar10_resnet50_bn_cifar_1337_nat \
    --background-dataset cifar10 \
    --num-train-samples 50000 \
    --num-test-samples 10000 \
    --batch-size 200 \
    --attack-name pgd_l2 \
    --attack-criterion LatentL2 \
    --target-sampler LatentSampler \
    --store-gt network_output \
    --pgd-step-size 0.1 \
    --pgd-epsilon 1000 \
    --pgd-iterations 1000 \
    --postfix _non_robust_cifar

# Reconstruct the d_robust_CIFAR dataset
python3 extract_grad_datasets.py \
    --dataset cifar10 \
    --arch resnet50_bn_cifar \
    --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075509_644_cifar10_resnet50_bn_cifar_1337_adv_train_l2_0_5 \
    --background-dataset cifar10 \
    --num-train-samples 50000 \
    --num-test-samples 10000 \
    --batch-size 200 \
    --attack-name pgd_l2 \
    --attack-criterion LatentL2 \
    --target-sampler LatentSampler \
    --store-gt network_output \
    --pgd-step-size 0.1 \
    --pgd-epsilon 1000 \
    --pgd-iterations 1000 \
    --postfix _robust_cifar

# Reconstruct ddet_cifar dataset
python3 extract_grad_datasets.py \
    --dataset cifar10 \
    --arch resnet50_bn_cifar \
    --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075452_407_cifar10_resnet50_bn_cifar_1337_nat \
    --background-dataset cifar10 \
    --num-train-samples 50000 \
    --num-test-samples 10000 \
    --batch-size 200 \
    --attack-name pgd_l2 \
    --attack-criterion LogitXent \
    --target-sampler OffsetOneHotSampler \
    --store-gt target \
    --pgd-step-size 0.1 \
    --pgd-epsilon 0.5 \
    --pgd-iterations 100 \
    --postfix _ddet_cifar

# Reconstruct drand_cifar dataset
python3 extract_grad_datasets.py \
    --dataset cifar10 \
    --arch resnet50_bn_cifar \
    --model-path /workspace/Projects/research-collections/code/models/fnb/20210906_075452_407_cifar10_resnet50_bn_cifar_1337_nat \
    --background-dataset cifar10 \
    --num-train-samples 50000 \
    --num-test-samples 10000 \
    --batch-size 200 \
    --attack-name pgd_l2 \
    --attack-criterion LogitXent \
    --target-sampler RandomOneHotSampler \
    --store-gt target \
    --pgd-step-size 0.1 \
    --pgd-epsilon 0.5 \
    --pgd-iterations 100 \
    --postfix _drand_cifar