import os, sys, time, random
import torch
import argparse
import torch.backends.cudnn as cudnn

import utils.loss
from utils.logger import get_logger
from utils.data import get_data_specs, get_data, get_transforms
from utils.network import get_network
from utils.training import validate
from utils.file import get_timestamp
from utils.attack import get_attack

from config.config import MODEL_PATH


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluates a neural network")
    # Model parameters
    parser.add_argument("--dataset", required=True, help="Training dataset")
    parser.add_argument("--arch", required=True, help="Model architecture")
    parser.add_argument("--model-path", type=str, default=None, help="Path to the model folder")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint.pth", help="Name of the checkpoint to load")
    # Others
    parser.add_argument("--seed", type=int, default=1337, help="Seed used")
    parser.add_argument("--workers", type=int, default=6, help="Number of data loading workers")
    # Evaluation parameters
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
    parser.add_argument("--augmentation", type=eval, default="True", choices=[True, False], help="Determines if augmentation is used")
    parser.add_argument("--evaluation-dataset", required=True, help="Evaluation dataset")
    parser.add_argument("--severity", type=int, help="Severity of corruption (for ImageNet-C datasets)")
    parser.add_argument("--criterion", default="Xent")
    # Evaluation under adversarial attack
    parser.add_argument("--attack-name", default=None, help="Evaluate under attack")
    parser.add_argument("--attack-criterion", default=None, help="Criterion used by attack")
    # PGD parameters
    parser.add_argument("--pgd-step-size", type=eval, help="PGD: Step size")
    parser.add_argument("--pgd-epsilon", type=eval, help="PGD: Epsilon")
    parser.add_argument("--pgd-iterations", type=int, help="PGD: Iterations")
    parser.add_argument("--pgd-random-start", type=eval, default="False", choices=[True, False], help="PGD: Random start")
    # Path 
    parser.add_argument("--postfix", type=str, default="", help="Postfix to append to filename")

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()

    return args


def main():
    args = parse_arguments()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    
    if (args.model_path == None) and (args.dataset == 'imagenet'):
        folder_name = "{}_{}_{}".format(args.dataset, args.arch, args.seed)
        model_path = os.path.join(MODEL_PATH, folder_name)
    elif args.model_path == None:
        raise ValueError("Invalid combination: Datatset: {} Model Path: {}".format(args.dataset, args.model_path))
    else:
        model_path = args.model_path

    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    eval_folder = os.path.join(model_path, 'eval')
    if not os.path.isdir(eval_folder):
        os.makedirs(eval_folder)

    filename = get_timestamp() + "_" + str(args.attack_name) + args.postfix + ".txt"
    logger = get_logger(eval_folder, filename=filename)

    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        logger.info("{} : {}".format(key, value))

    num_classes, mean, std, img_size, num_channels = get_data_specs(dataset=args.evaluation_dataset)
    train_transform, test_transform = get_transforms(dataset=args.evaluation_dataset,
                                                    augmentation=args.augmentation)
    logger.info("Train transform: {}".format(train_transform))
    logger.info("Test transform: {}".format(test_transform))

    data_train, data_test = get_data(args.evaluation_dataset,
                                        train_transform=train_transform, 
                                        test_transform=test_transform,
                                        severity=args.severity)
    
    test_loader = torch.utils.data.DataLoader(data_test,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)
    ### Network ###
    net = get_network(args.arch, num_classes)
    net.eval()
    if args.use_cuda:
        net.cuda()

    # Get pretrained model checkpoint
    if (args.dataset != 'imagenet'):
        model_ckpt = os.path.join(model_path, args.checkpoint_name)
        logger.info("Loading checkpoint: {}".format(model_ckpt))
        # Load the model
        network_data = torch.load(model_ckpt)
        net.load_state_dict(network_data["state_dict"])
    elif (args.dataset == 'imagenet') and (args.model_path != None):
        model_ckpt = os.path.join(model_path, args.checkpoint_name)
        logger.info("Loading checkpoint: {}".format(model_ckpt))
        # Load the model
        network_data = torch.load(model_ckpt)
        net.load_state_dict(network_data["state_dict"])

    ### Criterion ###
    if args.criterion in utils.loss.__dict__:
        criterion = utils.loss.__dict__[args.criterion]()
    else:
        raise ValueError('Unknown criterion: {}'.format(args.criterion))
    
    ### Attack ### 
    if args.attack_name:
        if args.attack_criterion in utils.loss.__dict__:
            attack_criterion = utils.loss.__dict__[args.attack_criterion]()
        else:
            raise ValueError('Unknown attack criterion: {}'.format(args.attack_criterion))

        attack = get_attack(attack_name=args.attack_name,
                        net=net,
                        attack_criterion=attack_criterion,
                        mean=mean,
                        std=std,
                        pgd_step_size=args.pgd_step_size,
                        pgd_epsilon=args.pgd_epsilon,
                        pgd_iterations=args.pgd_iterations,
                        pgd_random_start=args.pgd_random_start)
    else:
        attack = None
    top1, top5, loss = validate(test_loader, net, criterion, attack=attack, use_cuda=args.use_cuda)
    

if __name__ == "__main__":
    main()
