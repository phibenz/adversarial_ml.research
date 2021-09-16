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
from utils.network import get_network, get_num_parameters, get_num_non_trainable_parameters, get_num_trainable_parameters
from utils.training import accuracy, validate
from utils.data import get_data, get_transforms


def parse_arguments():
    parser = argparse.ArgumentParser(description="Trains a neural network")
    # Standard parameters
    parser.add_argument("--dataset", required=True, help="Training dataset")
    parser.add_argument("--arch", required=True, help="Model architecture")
    parser.add_argument("--seed", type=int, default=1337, help="Seed used")
    parser.add_argument("--criterion", default="Xent", help="Training criterion")
    parser.add_argument("--augmentation", type=eval, default="True", choices=[True, False], help="Determines if data augmentation is used")
    # Adversarial Training parameters
    parser.add_argument("--adversarial-training", type=eval, default="False", choices=[True, False], help="Train adversarially")
    parser.add_argument("--attack-criterion", default=None, help="Criterion used by attack")
    parser.add_argument("--targeted-attack", type=eval, default="False", choices=[True, False], help="Use targeted attack ")
    parser.add_argument("--attack-name", default=None, help="Attack used")
    parser.add_argument("--pgd-step-size", type=float, help="PGD: Step size")
    parser.add_argument("--pgd-epsilon", type=float, help="PGD: Epsilon")
    parser.add_argument("--pgg-iterations", type=int, help="PGD: Epsilon")
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
    # Evaluation parameters
    # parser.add_argument("--eval-pgd-l2", type=eval, default="False", choices=[True, False], help="Evaluate pgd l2 attack after each epoch")
    # parser.add_argument("--eval-pgd-linf", type=eval, default="False", choices=[True, False], help="Evaluate pgd linf attack after each epoch")
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
                                    test_transform=test_transform)

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
    
    # TODO: Adversarial Training - Attacks

    ### Recorders ### 
    recorder = RecorderMeter(args.epochs)
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    stat_csvwriter, stat_csvfile = get_stats_recorder(model_path)
    
    logger.info("Validation before training")
    val_acc1, val_acc5, val_loss = validate(test_loader, net, criterion, use_cuda=args.use_cuda)
    
    # Main loop
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
            
            # TODO: Adversarial Training
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

        logger.info("+++ Validation on pretrained test dataset +++")
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

    stat_csvfile.close()

if __name__ == '__main__':
    main()
