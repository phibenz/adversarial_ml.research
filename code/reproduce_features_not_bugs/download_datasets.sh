# Include paths
source ./config/config.py

if [ ! -d $DATA_PATH ]; then
  echo "Creating Folder: $DATA_PATH"
  mkdir $DATA_PATH
fi

DOWNLOAD_FILENAME=fnb_datasets.tar
DOWNLOAD_PATH="$DATA_PATH/$DOWNLOAD_FILENAME"
if [ ! -f $DOWNLOAD_PATH ]; then
  curl andrewilyas.com/datasets.tar -L -o $DOWNLOAD_PATH
fi

if [ ! -d $FNB_DATASET_PATH ]; then
  tar -xvf $DOWNLOAD_PATH -C $DATA_PATH
  # Change the dataset name 
  mv "$DATA_PATH/release_datasets" $FNB_DATASET_PATH
fi
