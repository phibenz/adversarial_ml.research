import os, sys, shutil, random, argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import utils.loss
import utils.target_sampler
from utils.target_sampler import one_hot
from utils.logger import get_logger
from utils.data import get_data_specs, get_data, get_transforms
from utils.network import get_network, Normalize, UnNormalize
from utils.training import validate
from utils.file import get_timestamp
from utils.attack import get_attack
# from utils.target_samplers import Gt_One_Hot_Sampler, Random_One_Hot_Sampler, Offset_One_Hot_Sampler, Real_Logit_Sampler, Latent_Sampler

from config.config import MODEL_PATH

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extracts a dataset of gradient images from a model")
    # Model parameters
    parser.add_argument("--dataset", required=True, help="Training dataset")
    parser.add_argument("--arch", required=True, help="Model architecture")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model folder")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint.pth", help="Name of the checkpoint to load")
    # Others
    parser.add_argument("--seed", type=int, default=1337, help="Seed used")
    parser.add_argument("--workers", type=int, default=6, help="Number of data loading workers")
    # Parameters regarding gradient images
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
    parser.add_argument("--background-dataset", default=None, help="Dataset used as background")
    parser.add_argument("--background-grad-imgs-path", default=None, help="Path to background images")
    parser.add_argument("--augmentation", type=eval, default="False", choices=[True, False], help="Determines if augmentation is used")
    parser.add_argument("--target-sampler", default="random_one_hot", help="Method to sample the logits")
    parser.add_argument("--store-gt", default="target", help="Logit (One-hot) value to store as the gt of the grad images")
    parser.add_argument("--num-train-samples", type=int, required=True, help="Number of train samples to generate")
    parser.add_argument("--num-test-samples", type=int, required=True, help="Number of test samples to generate")
    # Attack used for extraction
    parser.add_argument("--attack-name", default=None, help="Evaluate under attack")
    parser.add_argument("--attack-criterion", default=None, help="Criterion used by attack")
    # PGD parameters
    parser.add_argument("--pgd-step-size", type=eval, help="PGD: Step size")
    parser.add_argument("--pgd-epsilon", type=eval, help="PGD: Epsilon")
    parser.add_argument("--pgd-iterations", type=int, help="PGD: Iterations")
    parser.add_argument("--pgd-random-start", type=eval, default="False", choices=[True, False], help="PGD: Random start")
    # Path
    parser.add_argument("--postfix", type=str, default="", help="Postfix to append to the grad imgs folder name")
    # parser.add_argument("--adversarially_trained", type=eval, default="False", choices=[True, False], help="Train adversarially (default: False)")
    
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

    # Store the dataset in the folder of the model it was generated from 
    grad_imgs_path = os.path.join(args.model_path, 'grad_datasets')
    grad_imgs_folder_name = get_timestamp() + args.postfix
    grad_imgs_path = os.path.join(grad_imgs_path, grad_imgs_folder_name)
    os.makedirs(grad_imgs_path)

    logger = get_logger(grad_imgs_path)

    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        logger.info("{} : {}".format(key, value))

    # Ddataset used train the network, thus used as reference
    num_classes, _, _, _, _ = get_data_specs(dataset=args.dataset)
    # Background dataset
    _, mean, std, img_size, num_channels = get_data_specs(dataset=args.background_dataset)
    norm = Normalize(mean, std)
    unnorm = UnNormalize(mean, std)
    train_transform, test_transform  = get_transforms(dataset=args.dataset, 
                                                        augmentation=False)
    logger.info("Train transform: {}".format(train_transform ))
    logger.info("Test transform: {}".format(test_transform ))

    data_train, data_test = get_data(args.dataset,
                                    train_transform=train_transform,
                                    test_transform=test_transform)
    
    data_train_loader = torch.utils.data.DataLoader(data_train,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)
    data_test_loader = torch.utils.data.DataLoader(data_test,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    background_train_transform, background_test_transform = get_transforms(dataset=args.background_dataset, 
                                                                            augmentation=args.augmentation)
    logger.info("Background train transform: {}".format(background_train_transform))
    logger.info("Background test transform: {}".format(background_test_transform))

    background_data_train, background_data_test = get_data(args.background_dataset,
                                                                train_transform=background_train_transform,
                                                                test_transform=background_test_transform,
                                                                grad_imgs_path=args.background_grad_imgs_path)
    
    background_data_train_loader = torch.utils.data.DataLoader(background_data_train,
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            num_workers=args.workers,
                                                            pin_memory=True)
    background_data_test_loader = torch.utils.data.DataLoader(background_data_test,
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
    if (args.dataset != "imagenet"):
        model_ckpt = os.path.join(args.model_path, args.checkpoint_name)
        logger.info("Loading checkpoint: {}".format(model_ckpt))
        # Load the model
        network_data = torch.load(model_ckpt)
        net.load_state_dict(network_data["state_dict"])
    elif (args.dataset == 'imagenet') and (args.model_path != None):
        model_ckpt = os.path.join(args.model_path, args.checkpoint_name)
        logger.info("Loading checkpoint: {}".format(model_ckpt))
        # Load the model
        network_data = torch.load(model_ckpt)
        net.load_state_dict(network_data["state_dict"])
    
    ### Criterion ###
    if args.attack_criterion in utils.loss.__dict__:
        attack_criterion = utils.loss.__dict__[args.attack_criterion]()
    else:
        raise ValueError('Unknown attack criterion: {}'.format(args.attack_criterion))
    
    # Get the attack
    attack = get_attack(attack_name=args.attack_name,
                            net=net,
                            attack_criterion=attack_criterion,
                            mean=mean,
                            std=std,
                            pgd_step_size=args.pgd_step_size,
                            pgd_epsilon=args.pgd_epsilon,
                            pgd_iterations=args.pgd_iterations,
                            pgd_random_start=args.pgd_random_start,
                            latent_loss=attack_criterion.latent_loss)

    if args.target_sampler in utils.target_sampler.__dict__:
        target_sampler = utils.target_sampler.__dict__[args.target_sampler](num_classes=num_classes, 
                                                                            offset=1, 
                                                                            model=net)
    else:
        raise ValueError('Unknown target sampler: {}'.format(args.target_sampler))
    
    #########################################
    ######## Extracting the datasets ########
    #########################################

    num_samples_list = [args.num_train_samples, args.num_test_samples]
    background_loader_list = [background_data_train_loader, background_data_test_loader]
    reference_loader_list = [data_train_loader, data_test_loader]
    file_paths_list = [(os.path.join(grad_imgs_path, "x_train.pt"), os.path.join(grad_imgs_path, "y_train.pt")), 
                       (os.path.join(grad_imgs_path, "x_test.pt"), os.path.join(grad_imgs_path, "y_test.pt"))]

    for num_samples, background_loader, reference_loader, file_paths in zip(num_samples_list, background_loader_list, reference_loader_list, file_paths_list):

        # Prepare empty datasets
        x_data = torch.zeros(size=(num_samples, num_channels, img_size, img_size))
        y_data = torch.zeros(size=(num_samples, num_classes))
        
        # Iterators
        background_iterator = iter(background_loader)
        reference_iterator = iter(reference_loader)
        
        generated_samples = 0
        while generated_samples < num_samples:
            print("Generating samples {}/{}".format(generated_samples, num_samples))
            
            same_length = False
            while (not same_length):
                # The background image is the image set as starting point for the adversarial attack
                # The reference image is the image used to generate logits/targets
                try:
                    background_img, background_lbl = next(background_iterator)
                except StopIteration:
                    background_iterator = iter(background_loader)
                    background_img, background_lbl = next(background_iterator)
                
                try:
                    reference_img, reference_lbl = next(reference_iterator)
                except StopIteration:
                    reference_iterator = iter(reference_loader)
                    reference_img, reference_lbl = next(reference_iterator)
                if len(background_img) == len(reference_img):
                    same_length = True
            
            if args.use_cuda:
                background_img = background_img.cuda()
                background_lbl = background_lbl.cuda()
                reference_img = reference_img.cuda()
                reference_lbl = reference_lbl.cuda()
            
            target_label = target_sampler.sample(reference_img, reference_lbl, background_lbl)
            if args.use_cuda:
                target_label = target_label.cuda()
            
            x_adv = attack.run(background_img, target_label)
            target_label_net = net(x_adv)
            
            #### Swap target label and target_label_net as needed
            if generated_samples + len(x_adv) >= num_samples:
                x_data[generated_samples:generated_samples+len(x_adv)] = unnorm(x_adv[:num_samples-generated_samples]).cpu().detach()
                if args.store_gt == "target":
                    y_data[generated_samples:generated_samples+len(x_adv)] = target_label[:num_samples-generated_samples].cpu().detach()
                elif args.store_gt == "network_output":
                    y_data[generated_samples:generated_samples+len(x_adv)] = target_label_net[:num_samples-generated_samples].cpu().detach()
                elif args.store_gt == "background_gt":
                    bg_one_hot = one_hot(background_lbl.cpu().detach())
                    y_data[generated_samples:generated_samples+len(x_adv)] = bg_one_hot[:num_samples-generated_samples].cpu().detach()
            else:
                x_data[generated_samples:generated_samples+len(x_adv)] = unnorm(x_adv).cpu().detach()
                if args.store_gt == "target":
                    y_data[generated_samples:generated_samples+len(x_adv)] = target_label.cpu().detach()
                elif args.store_gt == "network_output":
                    y_data[generated_samples:generated_samples+len(x_adv)] = target_label_net.cpu().detach()
                elif args.store_gt == "background_gt":
                    bg_one_hot = one_hot(background_lbl.cpu().detach())
                    y_data[generated_samples:generated_samples+len(x_adv)] = bg_one_hot.cpu().detach()
            generated_samples += len(x_adv)
    
        torch.save(x_data, file_paths[0])
        torch.save(y_data, file_paths[1])
    #########################################

    if not args.dataset.startswith("grad_imgs"):
        if args.dataset == "mnist":
            grad_img_dataset = "grad_imgs_mnist"
        elif args.dataset == "cifar10":
            grad_img_dataset = "grad_imgs_cifar10"
        else:
            raise NotImplementedError
    # else:
    #     grad_img_dataset = args.dataset

    grad_img_train_transform, grad_img_test_transform = get_transforms(dataset=grad_img_dataset, 
                                                                        augmentation=False)


    grad_img_data_train, grad_img_data_test = get_data(grad_img_dataset,
                                                        train_transform=grad_img_train_transform,
                                                        test_transform=grad_img_test_transform,
                                                        grad_imgs_path=grad_imgs_path)
    
    grad_img_data_train_loader = torch.utils.data.DataLoader(grad_img_data_train,
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            num_workers=args.workers,
                                                            pin_memory=True)
    grad_img_data_test_loader = torch.utils.data.DataLoader(grad_img_data_test,
                                                            batch_size=args.batch_size,
                                                            shuffle=False,
                                                            num_workers=args.workers,
                                                            pin_memory=True)
    # Evaluate
    criterion_logit_xent = utils.loss.__dict__["LogitXent"]()
    logger.info("Evaluation on train dataset:")
    validate(grad_img_data_train_loader, net, criterion_logit_xent, use_cuda=args.use_cuda)
    logger.info("Evaluation on test dataset:")
    validate(grad_img_data_test_loader, net, criterion_logit_xent, use_cuda=args.use_cuda)

if __name__ == '__main__':
    main()
