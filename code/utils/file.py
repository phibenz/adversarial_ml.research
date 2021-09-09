import os
from datetime import datetime
from config.config import MODEL_PATH


def get_timestamp():
    ISOTIMEFORMAT='%Y%m%d_%H%M%S_%f'
    timestamp = '{}'.format(datetime.utcnow().strftime( ISOTIMEFORMAT)[:-3])
    return timestamp

def get_model_path(dataset_name, arch, seed, adversarial_training=False, subfolder="", postfix=""):
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    timestamp = get_timestamp()
    model_id = "{}_{}_{}_{}".format(timestamp, dataset_name, arch, seed)
    if adversarial_training:
        model_id = model_id + "_adv_train"
    model_id = model_id + postfix
    model_path = os.path.join(MODEL_PATH, subfolder, model_id)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    return model_path