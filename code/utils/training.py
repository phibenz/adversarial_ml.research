import logging
import torch
from utils.logger import AverageMeter, accuracy


def validate(val_loader, model, criterion, attack=None, use_cuda=True):
    logger = logging.getLogger('logbuch')
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if use_cuda:
            target = target.cuda()
            input = input.cuda()
        if attack:
            input = attack.run(input, target)
        
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        if len(target.shape) > 1:
            target = torch.argmax(target, dim=-1)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

    logger.info('==Test== Prec@1 {top1.avg:.3f}\t'
                'Prec@5 {top5.avg:.3f}\t' 
                'Error@1 {error1:.3f}'.format(
                top1=top1, top5=top5, error1=100-top1.avg))
    return top1.avg, top5.avg, losses.avg
