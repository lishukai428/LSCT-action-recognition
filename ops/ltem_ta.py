import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

class LTEM(nn.Module):
    def __init__(self, reduction=16):
        super(LTEM, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv_temp = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn_temp = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.avg_diff = nn.AvgPool2d(kernel_size=2,stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 64 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64 // reduction, 64),
            nn.Sigmoid())

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x1, x2, x3, x4, x5 = x[:, 0:3, :, :], x[:, 3:6, :, :], x[:, 6:9, :, :], x[:, 9:12, :, :], x[:, 12:15, :, :]
        x_c5 = torch.cat((x2 - x1, x3 - x2, x4 - x3, x5 - x4), 1).view(-1, 12, x2.size()[2],
                                                                       x2.size()[3])  # [N, 12, H, W]

        # local time
        x_c5 = self.avg_diff(x_c5)
        x_c5 = self.conv_temp(x_c5)
        x_c5 = self.bn_temp(x_c5)
        x_c5 = self.relu(x_c5)

        n, c, _, _ = x_c5.size()

        x_c = self.avg_pool(x_c5).view(n, c)  # [32, 64]
        x_c = self.fc(x_c).view(n, c, 1, 1)  # [32, 64]


        x_c5 = x_c5 * x_c.expand_as(x_c5)
        #
        x_3 = self.conv1(x3)
        x_3 = self.bn1(x_3)
        x_3 = self.relu(x_3)

        # x = x_3 + x_3 * x_c
        #
        x_c5 = F.interpolate(x_c5, x_3.size()[2:])
        x = x_3 + x_c5

        return x


class TA(nn.Module):

    def __init__(self, net, kernel_size=3, n_segment=8, shift_div=8):
        super(TA, self).__init__()
        self.net = net
        #self.channel = channel
        self.in_channels = self.net.in_channels
        self.n_segment = n_segment
        self.fold = self.in_channels // shift_div

        self.ta_shift = nn.Conv1d(self.in_channels, self.in_channels,
                                kernel_size=3, padding=1, groups=self.in_channels,
                                bias=False)
        self.ta_shift.weight.requires_grad = True
        self.ta_shift.weight.data.zero_()
        self.ta_shift.weight.data[:self.fold, 0, 2] = 1 # shift left
        self.ta_shift.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
        if 2*self.fold < self.in_channels:
            self.ta_shift.weight.data[2 * self.fold:, 0, 1] = 1 # fixed

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(self.in_channels, self.in_channels//16, kernel_size, stride=1, bias=False,
                              padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(self.in_channels//16)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(self.in_channels//16, self.in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment   #

        # import pdb;
        # pdb.set_trace()
        x_shift = x.view(n_batch, self.n_segment, c, h, w)
        x_shift = x_shift.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, n_segment)
        x_shift = x_shift.contiguous().view(n_batch*h*w, c, self.n_segment)
        x_shift = self.ta_shift(x_shift)  # (n_batch*h*w, c, n_segment)
        x_shift = x_shift.view(n_batch, h, w, c, self.n_segment)
        x_shift = x_shift.permute([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
        x_shift = x_shift.contiguous().view(nt, c, h, w)

        # import pdb;
        # pdb.set_trace()
        y = self.avg_pool(x_shift)
        nt, c, h, w = y.size()
        y = y.view(-1, c, self.n_segment).contiguous()  # n, c, t
        y = self.conv(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv1(y)
        # y = y.view(nt, c, 1, 1)
        y = self.sigmoid(y)
        y = y.view(nt, c, 1, 1)
        y = y.expand_as(x_shift)
        y = x_shift + x_shift*y

        out = self.net(y)
        # out = self.net(x_shift)

        return out


class Shift(nn.Module):

    def __init__(self, net, n_segment=8, shift_div=8):
        super(Shift, self).__init__()
        self.net = net
        self.in_channels = self.net.in_channels
        self.n_segment = n_segment
        self.fold = self.in_channels // shift_div

        self.ta_shift = nn.Conv1d(self.in_channels, self.in_channels,
                                kernel_size=3, padding=1, groups=self.in_channels,
                                bias=False)
        self.ta_shift.weight.requires_grad = True
        self.ta_shift.weight.data.zero_()
        self.ta_shift.weight.data[:self.fold, 0, 2] = 1 # shift left
        self.ta_shift.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
        if 2*self.fold < self.in_channels:
            self.ta_shift.weight.data[2 * self.fold:, 0, 1] = 1 # fixed

    def forward(self, x):

        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment   #

        # import pdb;
        # pdb.set_trace()
        x_shift = x.view(n_batch, self.n_segment, c, h, w)
        x_shift = x_shift.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, n_segment)
        x_shift = x_shift.contiguous().view(n_batch*h*w, c, self.n_segment)
        x_shift = self.ta_shift(x_shift)  # (n_batch*h*w, c, n_segment)
        x_shift = x_shift.view(n_batch, h, w, c, self.n_segment)
        x_shift = x_shift.permute([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
        x_shift = x_shift.contiguous().view(nt, c, h, w)

        out = self.net(x_shift)

        return out

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        # import pdb;
        # pdb.set_trace()
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)



def make_temporal_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))    # stage神经网络的每一层

    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        if place == 'block':   # In-place TSM
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())  # 提取神经网络的各个层
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i] = TA(net.conv1, n_segment=this_segment)   # b网络
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

        elif 'blockres' in place:   # Residual TSM
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = TA(b.conv1, n_segment=this_segment)
                return nn.Sequential(*blocks)

            # def make_shift(stage, this_segment):
            #     blocks = list(stage.children())
            #     print('=>add shift stage with {} blocks residual'.format(len(blocks)))
            #     for i, b in enumerate(blocks):
            #         if i % n_round == 0:
            #             blocks[i].conv1 = Shift(b.conv1, n_segment)
            #     return nn.Sequential(*blocks)
            def make_shift(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*blocks)

            net.conv1 = LTEM()
            net.bn1 = nn.Sequential()
            net.relu = nn.Sequential()
            # net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            # net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            # net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            # net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
            net.layer1 = make_shift(net.layer1, n_segment_list[0])
            net.layer2 = make_shift(net.layer2, n_segment_list[1])
            net.layer3 = make_shift(net.layer3, n_segment_list[2])
            net.layer4 = make_shift(net.layer4, n_segment_list[3])

    else:
        raise NotImplementedError(place)





class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding =1, bias=False)
        self.bn = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out










