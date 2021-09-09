import os, sys, random
import torch
import argparse
import dill
import torch.backends.cudnn as cudnn

from utils.file import get_model_path
from utils.data import get_data_specs
from utils.network import get_network

def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a neural network')
    # Checkpoint
    parser.add_argument('--checkpoint-pth', help='Path to the checkpoint to convert.')
    # pretrained
    parser.add_argument("--dataset", required=True, help="Training dataset")
    parser.add_argument("--arch", required=True, help="Model architecture")
    parser.add_argument("--seed", type=int, default=1337, help="Seed used")
    # Adversarially trained model
    parser.add_argument("--adversarial-training", type=eval, default="False", choices=[True, False], help="Train adversarially")
    # Folder structure
    parser.add_argument("--subfolder" ,type=str, default="", help="Subfolder to store results in")
    parser.add_argument("--postfix", type=str, default="", help="Attach postfix to model name")
    
    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    
    num_classes, _, _, _, _ = get_data_specs(dataset=args.dataset)
    
    model_path = get_model_path(dataset_name=args.dataset,
                                arch=args.arch,
                                seed=args.seed,
                                adversarial_training=args.adversarial_training,
                                subfolder=args.subfolder,
                                postfix=args.postfix)

    print('save path : {}'.format(model_path))
    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        print("{} : {}".format(key, value))

    print("=> Creating model '{}'".format(args.arch))
    net = get_network(args.arch, num_classes)
    net_state_dict = net.state_dict()
    
    download_checkpoint = torch.load(args.checkpoint_pth, pickle_module=dill)
    state_dict_path = 'model'
    if not ('model' in download_checkpoint):
        state_dict_path = 'state_dict'
    download_state_dict = download_checkpoint[state_dict_path]
    download_state_dict = {k[len('module.'):]:v for k,v in download_state_dict.items()}
    
    for net_key in net_state_dict.keys():
        match=0
        for download_key in download_state_dict.keys():
            if (net_key in download_key) and (not download_key.startswith("attacker")):
                net_state_dict[net_key] = download_state_dict[download_key]
                match=1
                break
        if match==0:
            print(">> WARNING!!! Could not find parameters for: {} ...".format(net_key))
    
    net.load_state_dict(net_state_dict)
    
    save_path = os.path.join(model_path, 'checkpoint.pth')
    print("Saving converted model to: {}".format(save_path))
    
    torch.save(
    {
      'arch'        : args.arch,
      'state_dict'  : net.state_dict(),
    }, 
    save_path)
    
if __name__ == '__main__':
    main()
