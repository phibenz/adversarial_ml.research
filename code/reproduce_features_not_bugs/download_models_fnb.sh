# Include paths
source ../config/config.py
source ./config/config.py

if [ ! -f $CHECKPOINT_PATH ]; then
  echo "Creating Folder: $CHECKPOINT_PATH"
  mkdir $CHECKPOINT_PATH
fi

FNB_MODEL_PATH="${CHECKPOINT_PATH}/fnb"
mkdir -p $FNB_MODEL_PATH

################################
########## CIFAR-10 ############
################################

# Download the ResNet50 base checkpoint
DOWNLOAD_FILENAME=cifar_nat.pt
DOWNLOAD_PATH="${FNB_MODEL_PATH}/$DOWNLOAD_FILENAME"
echo $DOWNLOAD_PATH
if [ ! -f $DOWNLOAD_PATH ]; then
  curl https://www.dropbox.com/s/yhpp4yws7sgi6lj/cifar_nat.pt?dl=1 -L -o $DOWNLOAD_PATH
fi

### L-2 model ###
# Epsilon 0.25
DOWNLOAD_FILENAME=cifar_l2_0_25.pt
DOWNLOAD_PATH="${FNB_MODEL_PATH}/$DOWNLOAD_FILENAME"
echo $DOWNLOAD_PATH
if [ ! -f $DOWNLOAD_PATH ]; then
  curl https://www.dropbox.com/s/2qsp7pt6t7uo71w/cifar_l2_0_25.pt?dl=1 -L -o $DOWNLOAD_PATH
fi

# Epsilon 0.5
DOWNLOAD_FILENAME=cifar_l2_0_5.pt
DOWNLOAD_PATH="${FNB_MODEL_PATH}/$DOWNLOAD_FILENAME"
echo $DOWNLOAD_PATH
if [ ! -f $DOWNLOAD_PATH ]; then
  curl https://www.dropbox.com/s/1zazwjfzee7c8i4/cifar_l2_0_5.pt?dl=1 -L -o $DOWNLOAD_PATH
fi

# Epsilon 1.0
DOWNLOAD_FILENAME=cifar_l2_1_0.pt
DOWNLOAD_PATH="${FNB_MODEL_PATH}/$DOWNLOAD_FILENAME"
echo $DOWNLOAD_PATH
if [ ! -f $DOWNLOAD_PATH ]; then
  curl https://www.dropbox.com/s/s2x7thisiqxz095/cifar_l2_1_0.pt?dl=1 -L -o $DOWNLOAD_PATH
fi

### L-infinity model ###
# Epsilon 8/255
DOWNLOAD_FILENAME=cifar_linf_8.pt
DOWNLOAD_PATH="${FNB_MODEL_PATH}/$DOWNLOAD_FILENAME"
echo $DOWNLOAD_PATH
if [ ! -f $DOWNLOAD_PATH ]; then
  curl https://www.dropbox.com/s/s2x7thisiqxz095/cifar_l2_1_0.pt?dl=1 -L -o $DOWNLOAD_PATH
fi

################################
########## ImageNet ############
################################
# L2 model
DOWNLOAD_FILENAME=imagenet_l2_3_0.pt
DOWNLOAD_PATH="${FNB_MODEL_PATH}/$DOWNLOAD_FILENAME"
echo $DOWNLOAD_PATH
if [ ! -f $DOWNLOAD_PATH ]; then
  curl https://www.dropbox.com/s/knf4uimlqsi1yz8/imagenet_l2_3_0.pt?dl=1 -L -o $DOWNLOAD_PATH
fi

# Linf model
DOWNLOAD_FILENAME=imagenet_linf_4.pt
DOWNLOAD_PATH="${FNB_MODEL_PATH}/$DOWNLOAD_FILENAME"
echo $DOWNLOAD_PATH
if [ ! -f $DOWNLOAD_PATH ]; then
  curl https://www.dropbox.com/s/axfuary2w1cnyrg/imagenet_linf_4.pt?dl=1 -L -o $DOWNLOAD_PATH
fi

DOWNLOAD_FILENAME=imagenet_linf_8.pt
DOWNLOAD_PATH="${FNB_MODEL_PATH}/$DOWNLOAD_FILENAME"
echo $DOWNLOAD_PATH
if [ ! -f $DOWNLOAD_PATH ]; then
  curl https://www.dropbox.com/s/yxn15a9zklz3s8q/imagenet_linf_8.pt?dl=1 -L -o $DOWNLOAD_PATH
fi