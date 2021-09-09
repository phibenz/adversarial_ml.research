'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.cifar.custom_modules import Identity, FakeReLU, SequentialWithArgs


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, normalization="bn", activation="relu"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if normalization == "bn":
            self.bn1 = nn.BatchNorm2d(planes)
        elif normalization == "id":
            self.bn1 = Identity()
        elif normalization == "gn":
            self.bn1 = nn.GroupNorm(num_groups=32, num_channels=planes)
        elif normalization == "ln":
            self.bn1 = nn.GroupNorm(num_groups=1, num_channels=planes)
        elif normalization == "in":
            self.bn1 = nn.InstanceNorm2d(planes, affine=True)
        else:
            raise ValueError()
            
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if normalization == "bn":
            self.bn2 = nn.BatchNorm2d(planes)
        elif normalization == "id":
            self.bn2 = Identity()
        elif normalization == "gn":
            self.bn2 = nn.GroupNorm(num_groups=32, num_channels=planes)
        elif normalization == "ln":
            self.bn2 = nn.GroupNorm(num_groups=1, num_channels=planes)
        elif normalization == "in":
            self.bn2 = nn.InstanceNorm2d(planes, affine=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if normalization == "bn":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            elif normalization == "id":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
                )
            elif normalization == "gn":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(num_groups=32, num_channels=self.expansion*planes)
                )
            elif normalization == "ln":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(num_groups=1, num_channels=self.expansion*planes)
                )
            elif normalization == "in":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.InstanceNorm2d(self.expansion*planes, affine=True)
                )

        self.activation = activation
        if activation == "relu":
            self.activation_fn = F.relu
        elif activation == "selu":
            self.activation_fn = F.selu
        elif activation == "sigmoid":
            self.activation_fn = F.sigmoid
        elif activation == "tanh":
            self.activation_fn = F.tanh
        elif activation == "leaky_relu":
            self.activation_fn = F.leaky_relu
        else:
            raise ValueError

    def forward(self, x, fake_relu=False):
        out = self.activation_fn(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if fake_relu:
            out = FakeReLU.apply(out)
        else:
            out = self.activation_fn(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, normalization="bn", activation="relu"):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        if normalization == "bn":
            self.bn1 = nn.BatchNorm2d(planes)
        elif normalization == "id":
            self.bn1 = Identity()
        elif normalization == "gn":
            self.bn1 = nn.GroupNorm(num_groups=32, num_channels=planes)
        elif normalization == "ln":
            self.bn1 = nn.GroupNorm(num_groups=1, num_channels=planes)
        elif normalization == "in":
            self.bn1 = nn.InstanceNorm2d(planes, affine=True)
        else:
            raise ValueError()
            
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if normalization == "bn":
            self.bn2 = nn.BatchNorm2d(planes)
        elif normalization == "id":
            self.bn2 = Identity()
        elif normalization == "gn":
            self.bn2 = nn.GroupNorm(num_groups=32, num_channels=planes)
        elif normalization == "ln":
            self.bn2 = nn.GroupNorm(num_groups=1, num_channels=planes)
        elif normalization == "in":
            self.bn2 = nn.InstanceNorm2d(planes, affine=True)
        
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        if normalization == "bn":
            self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        elif normalization == "id":
            self.bn3 = Identity()
        elif normalization == "gn":
            self.bn3 = nn.GroupNorm(num_groups=32, num_channels=self.expansion*planes)
        elif normalization == "ln":
            self.bn3 = nn.GroupNorm(num_groups=1, num_channels=self.expansion*planes)
        elif normalization == "in":
            self.bn3 = nn.InstanceNorm2d(self.expansion*planes, affine=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if normalization == "bn":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            elif normalization == "id":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
                )
            elif normalization == "gn":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(num_groups=32, num_channels=self.expansion*planes)
                )
            elif normalization == "ln":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(num_groups=1, num_channels=self.expansion*planes)
                )
            elif normalization == "in":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.InstanceNorm2d(self.expansion*planes, affine=True)
                )
        self.activation = activation
        if activation == "relu":
            self.activation_fn = F.relu
        elif activation == "selu":
            self.activation_fn = F.selu
        elif activation == "sigmoid":
            self.activation_fn = F.sigmoid
        elif activation == "tanh":
            self.activation_fn = F.tanh
        elif activation == "leaky_relu":
            self.activation_fn = F.leaky_relu
        else:
            raise ValueError

    def forward(self, x, fake_relu=False):
        out = self.activation_fn(self.bn1(self.conv1(x)))
        out = self.activation_fn(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if fake_relu:
            out = FakeReLU.apply(out)
        else:
            out = self.activation_fn(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, normalization="bn", activation="relu"):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.normalization = normalization
        self.activation = activation

        if activation == "relu":
            self.activation_fn = F.relu
        elif activation == "selu":
            self.activation_fn = F.selu
        elif activation == "sigmoid":
            self.activation_fn = F.sigmoid
        elif activation == "tanh":
            self.activation_fn = F.tanh
        elif activation == "leaky_relu":
            self.activation_fn = F.leaky_relu
        else:
            raise ValueError
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if normalization == "bn":
            self.bn1 = nn.BatchNorm2d(64)
        elif normalization == "id":
            self.bn1 = Identity()
        elif normalization == "gn":
            self.bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
        elif normalization == "ln":
            self.bn1 = nn.GroupNorm(num_groups=1, num_channels=64)
        elif normalization == "in":
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
        else:
            raise ValueError()
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.normalization, self.activation))
            self.in_planes = planes * block.expansion
        return SequentialWithArgs(*layers)

    def forward(self, x, with_latent=False, fake_relu=False):
        out = self.activation_fn(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out, fake_relu=fake_relu)
        out = F.avg_pool2d(out, 4)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        if with_latent:
            return pre_out
        return final


def ResNet18(normalization="bn", num_classes=10, activation="relu"):
    return ResNet(BasicBlock, [2,2,2,2], normalization=normalization, num_classes=num_classes, activation=activation)

def ResNet34(normalization="bn", num_classes=10, activation="relu"):
    return ResNet(BasicBlock, [3,4,6,3], normalization=normalization, num_classes=num_classes, activation=activation)

def ResNet50(normalization="bn", num_classes=10, activation="relu"):
    return ResNet(Bottleneck, [3,4,6,3], normalization=normalization, num_classes=num_classes, activation=activation)

def ResNet101(normalization="bn", num_classes=10, activation="relu"):
    return ResNet(Bottleneck, [3,4,23,3], normalization=normalization, num_classes=num_classes, activation=activation)

def ResNet152(normalization="bn", num_classes=10, activation="relu"):
    return ResNet(Bottleneck, [3,8,36,3], normalization=normalization, num_classes=num_classes, activation=activation)
