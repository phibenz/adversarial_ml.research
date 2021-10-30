import os, sys, shutil, time, random, copy, csv
import argparse

import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn

import utils.loss
from utils.file import get_model_path
from utils.logger import get_logger, get_stats_recorder, AverageMeter, RecorderMeter
from utils.data import get_data_specs, get_transforms
from utils.network import get_network, get_num_parameters, get_num_non_trainable_parameters, get_num_trainable_parameters, Normalize, UnNormalize, get_input_grad
from utils.training import accuracy, validate
from utils.data import get_data, get_transforms
from utils.attack import get_attack

def parse_arguments():
    parser = argparse.ArgumentParser(description="Trains a neural network")
    # Standard parameters
    parser.add_argument("--dataset", required=True, help="Training dataset")
    parser.add_argument("--arch", required=True, help="Model architecture")
    parser.add_argument("--seed", type=int, default=1337, help="Seed used")
    parser.add_argument("--criterion", default="Xent", help="Training criterion")
    parser.add_argument("--augmentation", type=eval, default="True", choices=[True, False], help="Determines if data augmentation is used")
    parser.add_argument("--grad-imgs-path", type=str, default=None, help="Path to the gradient images")
    parser.add_argument("--evaluation-dataset", default=None, help="Evaluation dataset different from dataset")
    # Adversarial Training parameters
    parser.add_argument("--adversarial-training", type=eval, default="False", choices=[True, False], help="Train adversarially")
    parser.add_argument("--attack-criterion", default=None, help="Criterion used by attack")
    parser.add_argument("--targeted-attack", type=eval, default="False", choices=[True, False], help="Use targeted attack ")
    parser.add_argument("--attack-name", default=None, help="Attack used")
    parser.add_argument("--pgd-step-size", type=float, help="PGD: Step size")
    parser.add_argument("--pgd-epsilon", type=float, help="PGD: Epsilon")
    parser.add_argument("--pgd-iterations", type=int, help="PGD: Epsilon")
    parser.add_argument("--pgd-random-start", type=eval, default="False", choices=[True, False], help="PGD: Random start")
    # Optimization options
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
    parser.add_argument("--optimizer", default="sgd", help="Used optimizer")
    parser.add_argument("--learning-rate", type=float, required=True, help="Learning Rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay (L2 penalty)")
    parser.add_argument("--schedule", type=int, nargs="+", default=[], help="Decrease learning rate at these epochs")
    parser.add_argument("--gamma", type=float, default=0.1, help="LR is multiplied by gamma on schedule")
    # LIGS
    parser.add_argument("--record-ligs", type=eval, default="False", choices=[True, False], help="Record LIGS")
    # PGD Eval
    parser.add_argument("--eval-pgd-l2", type=eval, default="False", choices=[True, False], help="Evaluate pgd l2 attack after each epoch")
    parser.add_argument("--eval-pgd-linf", type=eval, default="False", choices=[True, False], help="Evaluate pgd linf attack after each epoch")
    # Shifted label evaluation
    parser.add_argument("--eval-shifted-label", type=eval, default="False", choices=[True, False], help="Evaluate on shifted labels")
    # Folder structure
    parser.add_argument("--subfolder" ,type=str, default="", help="Subfolder to store results in")
    parser.add_argument("--postfix", type=str, default="", help="Attach postfix to model name")
    # MISC
    parser.add_argument("--workers", type=int, default=6, help="Number of data loading workers")
    parser.add_argument("--print-freq", default=200, type=int, help="print frequency")

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()

    return args


def main():
    args = parse_arguments()

    # Setting the seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # Get the path where the model will be stored
    model_path = get_model_path(dataset_name=args.dataset,
                                arch=args.arch,
                                seed=args.seed,
                                adversarial_training=args.adversarial_training,
                                subfolder=args.subfolder, 
                                postfix=args.postfix)
    
    # Logging
    logger = get_logger(model_path)
    
    logger.info("Python version : {}".format(sys.version.replace('\n', ' ')))
    logger.info("Torch version : {}".format(torch.__version__))
    logger.info("Cudnn version : {}".format(torch.backends.cudnn.version()))
    logger.info("Model Path : {}".format(model_path))
    
    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        logger.info("{} : {}".format(key, value))
    
    # Save this file
    file_save_path = os.path.join(model_path, 'code')
    if not os.path.isdir(file_save_path):
        os.makedirs(file_save_path)
    shutil.copy(sys.argv[0], os.path.join(file_save_path, sys.argv[0]))

    num_classes, mean, std, img_size, num_channels = get_data_specs(dataset=args.dataset)
    train_transform, test_transform = get_transforms(dataset=args.dataset, augmentation=args.augmentation)

    logger.info("Train transform: {}".format(train_transform))
    logger.info("Test transform: {}".format(test_transform))
    
    data_train, data_test = get_data(args.dataset,
                                    train_transform=train_transform,
                                    test_transform=test_transform,
                                    grad_imgs_path=args.grad_imgs_path)

    train_loader = torch.utils.data.DataLoader(data_train,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.workers,
                                                pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)
    if args.evaluation_dataset:
        # Overwrite the test_loader with test set of evaluation dataset
        train_transform, test_transform = get_transforms(dataset=args.evaluation_dataset, augmentation=True)
        _, data_test = get_data(args.evaluation_dataset,
                                train_transform=train_transform,
                                test_transform=test_transform,
                                grad_imgs_path=args.grad_imgs_path)
        test_loader = torch.utils.data.DataLoader(data_test,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)

    if args.eval_shifted_label:
        # Copy of the test loader
        shifted_label_test_loader = copy.deepcopy(test_loader)
        shifted_label_test_loader.dataset.targets = [(i + 1) % num_classes for i in shifted_label_test_loader.dataset.targets]


    ### Network ###
    net = get_network(args.arch, num_classes)
    net.train()
    logger.info("Network :\n {}".format(net))
    non_trainale_params = get_num_non_trainable_parameters(net)
    trainale_params = get_num_trainable_parameters(net)
    total_params = get_num_parameters(net)
    logger.info("Trainable parameters: {}".format(trainale_params))
    logger.info("Non Trainable parameters: {}".format(non_trainale_params))
    logger.info("Total # parameters: {}".format(total_params))
    
    if args.use_cuda:
        net.cuda()

    ### Criterion ###
    if args.criterion in utils.loss.__dict__:
        criterion = utils.loss.__dict__[args.criterion]()
    else:
        raise ValueError('Unknown criterion: {}'.format(args.criterion))
    
    ### Optimizer ###
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params=net.parameters(), 
                                    lr=args.learning_rate, 
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay, 
                                    nesterov=False)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(params=net.parameters(), 
                                    lr=args.learning_rate, 
                                    weight_decay=args.weight_decay,
                                    amsgrad=False)
    else:
        raise ValueError('Unknown optimizer: {}'.format(args.optimizer))
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[i * len(train_loader) for i in args.schedule],
                                                    gamma=args.gamma)
    
    # Adversarial Training - Attack
    if args.adversarial_training:
        if args.attack_criterion in utils.loss.__dict__:
                attack_criterion = utils.loss.__dict__[args.attack_criterion]()
        else:
            raise ValueError('Unknown attack criterion: {}'.format(args.attack_criterion))

        adv_training_attack = get_attack(attack_name=args.attack_name,
                                        net=net,
                                        attack_criterion=attack_criterion,
                                        mean=mean,
                                        std=std,
                                        pgd_step_size=args.pgd_step_size,
                                        pgd_epsilon=args.pgd_epsilon,
                                        pgd_iterations=args.pgd_iterations,
                                        pgd_random_start=args.pgd_random_start)

    # Evaluation attacks
    if args.eval_pgd_l2 or args.eval_pgd_linf:
        pgd_attack_criterion = utils.loss.__dict__['NegXent']()
    if args.eval_pgd_l2:
        eval_pgd_l2_attack = get_attack(attack_name='pgd_l2',
                                        net=net,
                                        attack_criterion=pgd_attack_criterion,
                                        mean=mean,
                                        std=std,
                                        pgd_step_size=0.1,
                                        pgd_epsilon=0.25,
                                        pgd_iterations=10,
                                        pgd_random_start=False)
    
    if args.eval_pgd_linf:
        eval_pgd_linf_attack = get_attack(attack_name='pgd_linf',
                                        net=net, 
                                        attack_criterion=pgd_attack_criterion,
                                        mean=mean,
                                        std=std,
                                        pgd_step_size=0.001,
                                        pgd_epsilon=0.00392,
                                        pgd_iterations=10,
                                        pgd_random_start=False)

    ### Recorders ### 
    recorder = RecorderMeter(args.epochs)
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    ### Log & Stats Files ###
    stat_csvwriter, stat_csvfile = get_stats_recorder(model_path)
    if args.record_ligs:
        ligs_stats_f = open(os.path.join(model_path, "ligs.csv"), "a")
    if args.eval_pgd_l2:
        eval_pgd_l2_f = open(os.path.join(model_path, "pgd_l2.csv"), "a")
    if args.eval_pgd_linf:
        eval_pgd_linf_f = open(os.path.join(model_path, "pgd_linf.csv"), "a")
    if args.eval_shifted_label:
        eval_shifted_label_f = open(os.path.join(model_path, "shifted_label.csv"), "a")

    logger.info("Validation before training")
    val_acc1, val_acc5, val_loss = validate(test_loader, net, criterion, use_cuda=args.use_cuda)
    
    #######################
    #### Training Loop ####
    #######################
    start_time = time.time()
    epoch_time = AverageMeter()
    # previous_learning_rate = args.learning_rate
    for epoch in range(args.epochs):
        logger.info('[Epoch={:03d}/{:03d}]\t'.format(epoch, args.epochs) + \
                    '[Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)))

        losses.reset()
        top1.reset()
        top5.reset()
        
        # switch to train mode
        net.train()

        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            if args.use_cuda:
                target = target.cuda()
                input = input.cuda()
            
            if args.targeted_attack:
                adv_target = torch.randint_like(target, low=0, high=num_classes)
            else:
                adv_target = target
            
            if args.adversarial_training:
                input = adv_training_attack.run(input, adv_target)
            
            output = net(input)
            loss = criterion(output, target)
            
            # measure accuracy and record loss
            if len(target.shape) > 1:
                target = torch.argmax(target, dim=-1)
            
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # measure elapsed time
            end = time.time()

            if i % args.print_freq == 0:
                logger.info('Epoch: [{:03d}][{:03d}/{:03d}]\t'
                            'LR: {lr:5f}\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            epoch, i, len(train_loader), lr=optimizer.param_groups[0]["lr"], loss=losses, top1=top1, top5=top5))
        
        logger.info('==Train== Prec@1 {top1.avg:.3f}\t'
                    'Prec@5 {top5.avg:.3f}\t'
                    'Error@1 {error:.3f}'.format(
                    top1=top1, top5=top5, error=100-top1.avg))

        logger.info("+++ Validation on test dataset +++")
        val_acc1, val_acc5, val_loss = validate(val_loader=test_loader, 
                                                model=net, 
                                                criterion=criterion, 
                                                use_cuda=args.use_cuda)
        recorder.update(epoch, losses.avg, top1.avg, val_loss, val_acc1)

        # Writing to the stats file
        stat_csvwriter.writerow([epoch+1, top1.avg, top5.avg, losses.avg, val_acc1, val_acc5, val_loss])
        stat_csvfile.flush()
        
        save_path = os.path.join(model_path, 'checkpoint.pth')
        torch.save(
            {
          'arch'        : args.arch,
          'state_dict'  : net.state_dict(),
          'optimizer'   : optimizer.state_dict(),
            }, save_path)

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        #####################
        #### Record LIGS ####
        #####################
        if args.record_ligs:
            norm = Normalize(mean=mean, std=std)
            unnorm = UnNormalize(mean=mean, std=std)

            ligs_criterion = torch.nn.CrossEntropyLoss(reduction="none")
            if args.use_cuda:
                ligs_criterion = ligs_criterion.cuda()
            
            cos_sim = torch.nn.CosineSimilarity(dim=1, eps=0.)

            cos_sim_tensor_logit = torch.tensor([])
            cos_sim_tensor_grad = torch.tensor([])

            for img, lbl in test_loader:
                if args.use_cuda:
                    img, lbl = img.cuda(), lbl.cuda()
                batch_size, _, _, _ = img.shape
                
                # Get clean gradient
                out_clean, grad_clean = get_input_grad(model=net, 
                                                    img=img, 
                                                    lbl=lbl, 
                                                    eps=0.01,
                                                    norm=norm,
                                                    unnorm=unnorm,
                                                    delta_init='none', 
                                                    backprop=False, 
                                                    cuda=args.use_cuda)
                # Get perturbed gradient
                out_perturbed, grad_perturbed = get_input_grad(model=net, 
                                                                img=img, 
                                                                lbl=lbl, 
                                                                eps=0.01, 
                                                                norm=norm,
                                                                unnorm=unnorm,
                                                                delta_init='gaussian', 
                                                                backprop=False, 
                                                                cuda=args.use_cuda)

                sim_logit = cos_sim(out_clean, out_perturbed)
                sim_logit = sim_logit[~torch.isnan(sim_logit)]
                sim_logit = sim_logit[~torch.isinf(sim_logit)]
                cos_sim_tensor_logit = torch.cat([cos_sim_tensor_logit, sim_logit.cpu().detach()])

                sim_grad = cos_sim(grad_clean.reshape(batch_size, -1), grad_perturbed.reshape(batch_size, -1))
                sim_grad = sim_grad[~torch.isnan(sim_grad)]
                sim_grad = sim_grad[~torch.isinf(sim_grad)] 
                cos_sim_tensor_grad = torch.cat([cos_sim_tensor_grad, sim_grad.cpu().detach()])

            sim_logit, sim_grad = torch.mean(cos_sim_tensor_logit), torch.mean(cos_sim_tensor_grad)
            ligs_stats_f.write("{},{}\n".format(sim_logit, sim_grad))
            ligs_stats_f.flush()

        #################
        #### Eval PGD ###
        #################
        if args.eval_pgd_l2:
            val_acc1, val_acc5, val_loss = validate(val_loader=test_loader,
                                                    model=net,
                                                    criterion=pgd_attack_criterion,
                                                    attack=eval_pgd_l2_attack,
                                                    use_cuda=args.use_cuda)
            eval_pgd_l2_f.write("{},{},{}\n".format(val_acc1, val_acc5, val_loss))
            eval_pgd_l2_f.flush()

        if args.eval_pgd_linf:
            val_acc1, val_acc5, val_loss = validate(val_loader=test_loader,
                                                    model=net,
                                                    criterion=pgd_attack_criterion,
                                                    attack=eval_pgd_linf_attack,
                                                    use_cuda=args.use_cuda)
            eval_pgd_linf_f.write("{},{},{}\n".format(val_acc1, val_acc5, val_loss))
            eval_pgd_linf_f.flush()
        
        ###########################
        #### Eval Shifted Label ###
        ###########################
        if args.eval_shifted_label:
            val_acc1, val_acc5, val_loss = validate(val_loader=shifted_label_test_loader,
                                                    model=net,
                                                    criterion=criterion,
                                                    use_cuda=args.use_cuda)
            eval_shifted_label_f.write("{},{},{}\n".format(val_acc1, val_acc5, val_loss))
            eval_shifted_label_f.flush()

    stat_csvfile.close()
    if args.record_ligs:
        ligs_stats_f.close()
    if args.eval_pgd_l2:
        eval_pgd_l2_f.close()
    if args.eval_pgd_linf:
        eval_pgd_linf_f.close()
    if args.eval_shifted_label:
        eval_shifted_label_f.close()

if __name__ == '__main__':
    main()
