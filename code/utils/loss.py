from torch.nn.modules.loss import _WeightedLoss
from torch.nn.functional import log_softmax, softmax, cross_entropy


class Xent(_WeightedLoss):
    def __init__(self):
        super(Xent, self).__init__(weight=None, size_average=None, reduce=None, reduction='none')
    
    def forward(self, pred, target):
        return cross_entropy(pred, target)

class NegXent(_WeightedLoss):
    def __init__(self):
        super(NegXent, self).__init__(weight=None, size_average=None, reduce=None, reduction='none')
    
    def forward(self, pred, target):
        return -cross_entropy(pred, target)

class LogitL2(_WeightedLoss):
    def __init__(self):
        super(LogitL2, self).__init__(weight=None, size_average=None, reduce=None, reduction='none')

    def forward(self, pred_logit, target_logit):
        logit_gap = target_logit - pred_logit
        logit_l2 = logit_gap.norm(p=2, dim=1)
        loss = torch.mean(logit_l2)
        return loss
        

# def neg_log_xent(pred, target):
#     target = torch.argmax(target, dim=-1)
#     loss = -cross_entropy(pred, target)
#     return loss

# def logit_xent(pred, target):
#     target = torch.argmax(target, dim=-1)
#     loss = cross_entropy(pred, target)
#     return loss
    
# def latent_l2(pred, target):
#     # loss = torch.div(torch.norm(target - pred, dim=1), torch.norm(pred, dim=1))
#     loss = torch.norm(target - pred, p=2, dim=1)
#     loss = torch.mean(loss)
#     return loss

# TODO: Kwards inputs
class CW(_WeightedLoss):
    def __init__(self, num_classes, confidence=0.0, cuda=False):
        super(CW, self).__init__(weight=None, size_average=None, reduce=None, reduction='none')

        self.confidence = confidence
        self.num_classes = num_classes
        self.cuda = cuda

    def forward(self, logit, lbl):
        one_hot_labels = one_hot(lbl.cpu(), num_classes=self.num_classes)
        if self.cuda:
            one_hot_labels = one_hot_labels.cuda()

        logit_gt = (one_hot_labels * logit).sum(1)
        not_logit_gt = ((1. - one_hot_labels) * logit - one_hot_labels * 10000.).max(1)[0]
        loss = torch.clamp(logit_gt - not_logit_gt, min=-self.confidence)
        return torch.mean(loss)

