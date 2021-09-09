# Include paths
source ./config/config.py

FNB_MODEL_PATH="${CHECKPOINT_PATH}/fnb"

################################
########## CIFAR-10 ############
################################

# Download the ResNet50 base checkpoint
DOWNLOAD_FILENAME=cifar_nat.pt
DOWNLOAD_PATH="${FNB_MODEL_PATH}/$DOWNLOAD_FILENAME"
python3 convert_model_fnb.py \
        --checkpoint-pth $DOWNLOAD_PATH \
        --dataset cifar10 \
        --arch resnet50_bn_cifar \
        --subfolder fnb \
        --postfix _nat

### L-2 model ###
# Epsilon 0.25
DOWNLOAD_FILENAME=cifar_l2_0_25.pt
DOWNLOAD_PATH="${FNB_MODEL_PATH}/$DOWNLOAD_FILENAME"
python3 convert_model_fnb.py \
        --checkpoint-pth $DOWNLOAD_PATH \
        --dataset cifar10 \
        --arch resnet50_bn_cifar \
        --adversarial-training True \
        --subfolder fnb \
        --postfix _l2_0_25

# Epsilon 0.5
DOWNLOAD_FILENAME=cifar_l2_0_5.pt
DOWNLOAD_PATH="${FNB_MODEL_PATH}/$DOWNLOAD_FILENAME"
python3 convert_model_fnb.py \
        --checkpoint-pth $DOWNLOAD_PATH \
        --dataset cifar10 \
        --arch resnet50_bn_cifar \
        --adversarial-training True \
        --subfolder fnb \
        --postfix _l2_0_5

# Epsilon 1.0
DOWNLOAD_FILENAME=cifar_l2_1_0.pt
DOWNLOAD_PATH="${FNB_MODEL_PATH}/$DOWNLOAD_FILENAME"
python3 convert_model_fnb.py \
        --checkpoint-pth $DOWNLOAD_PATH \
        --dataset cifar10 \
        --arch resnet50_bn_cifar \
        --adversarial-training True \
        --subfolder fnb \
        --postfix _l2_1_0

### L-infinity model ###
# Epsilon 8/255
DOWNLOAD_FILENAME=cifar_linf_8.pt
DOWNLOAD_PATH="${FNB_MODEL_PATH}/$DOWNLOAD_FILENAME"
python3 convert_model_fnb.py \
        --checkpoint-pth $DOWNLOAD_PATH \
        --dataset cifar10 \
        --arch resnet50_bn_cifar \
        --adversarial-training True \
        --subfolder fnb \
        --postfix _linf_8


# ################################
# ########## ImageNet ############
# ################################
# L2 model
DOWNLOAD_FILENAME=imagenet_l2_3_0.pt
DOWNLOAD_PATH="${FNB_MODEL_PATH}/$DOWNLOAD_FILENAME"
python3 convert_model_fnb.py \
        --checkpoint-pth $DOWNLOAD_PATH \
        --dataset imagenet \
        --arch resnet50 \
        --adversarial-training True \
        --subfolder fnb \
        --postfix _l2_3_0

# Linf model
DOWNLOAD_FILENAME=imagenet_linf_4.pt
DOWNLOAD_PATH="${FNB_MODEL_PATH}/$DOWNLOAD_FILENAME"
python3 convert_model_fnb.py \
        --checkpoint-pth $DOWNLOAD_PATH \
        --dataset imagenet \
        --arch resnet50 \
        --adversarial-training True \
        --subfolder fnb \
        --postfix _linf_4


DOWNLOAD_FILENAME=imagenet_linf_8.pt
DOWNLOAD_PATH="${FNB_MODEL_PATH}/$DOWNLOAD_FILENAME"
python3 convert_model_fnb.py \
        --checkpoint-pth $DOWNLOAD_PATH \
        --dataset imagenet \
        --arch resnet50 \
        --adversarial-training True \
        --subfolder fnb \
        --postfix _linf_8

