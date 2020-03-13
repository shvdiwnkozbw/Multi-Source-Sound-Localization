import torch
import numpy as np
import torchvision
from itertools import zip_longest
from pathlib import Path
import torch.nn as nn


def crnn2(inputdim=64, outputdim=527, pretrained=True):
    model = CRNN(
        inputdim=inputdim,
        outputdim=outputdim,
        blocktype="ConvConvBlock",
        filter=[16, 64, 128, 128, 512, 512],
        filtersizes=[3, 3, 3, 3, 3, 3],
        pooling=[2, 2, 2, [1, 2], [1, 2], [1, 2]],
        pooltype="MaxPool2d",
        hidden_size=256,
        temppool='linear',
    )
    if pretrained:
        state = torch.load(Path(__file__).parent / 'crnn2.pth')
        model.load_state_dict(state, strict=False)
    return model


def mobilecrnn_v2(inputdim=64, outputdim=527, pretrained=True):
    ## mAP: 0.2736431837819642
    model = MobileCRNN(
        inputdim, outputdim, **{
            "filters": [64, 64, 128, 128, 256, 256, 512, 512],
            "kernels": [5, 3, 3, 3, 3, 3, 3, 3],
            "padding": [2, 1, 1, 1, 1, 1, 1, 1],
            "strides": [2, 1, 1, 1, 1, 1, 1, 1],
            "pooling": [[2], [1, 2], [1, 1], [1, 2], [1], [1, 2], [1], [1],
                        [1]]
        })
    if pretrained:
        state = torch.load(str(Path(__file__).parent / 'mobilecrnn_v2.pth'))
        try:
            model.load_state_dict(state, strict=False)
        except:
            state = {k: v for k, v in state.items() if 'outputlayer' not in k}
            model.load_state_dict(state, strict=False)
    return model


def mobilecrnn_v1(inputdim=64, outputdim=527, pretrained=True):
    # mAP: 0.2593157546937745
    model = MobileCRNN(inputdim, outputdim)
    if pretrained:
        state = torch.load(Path(__file__).parent / 'mobilecrnn_v1.pth')
        model.load_state_dict(state, strict=False)
    return model


def crnn1_linear(inputdim=64, outputdim=527, pretrained=True):
    model = CRNN(
        inputdim=inputdim,
        outputdim=outputdim,
        blocktype="StandardBlock",
        filter=[16, 64, 128, 128],
        filtersizes=[3, 3, 3, 3],
        pooling=[2, 2, [1, 2], [1, 2]],
        pooltype="MaxPool2d",
        hidden_size=128,
        temppool='attention',
    )
    if pretrained:
        state = torch.load(Path(__file__).parent / 'crnn1_linear.pth')
        model.load_state_dict(state, strict=False)
    return model


def crnn1_attention(inputdim=64, outputdim=527, pretrained=True):
    model = CRNN(
        inputdim=inputdim,
        outputdim=outputdim,
        blocktype="ConvConvBlock",
        filter=[16, 64, 128, 128],
        filtersizes=[3, 3, 3, 3],
        pooling=[2, 2, [1, 2], [1, 2]],
        pooltype="MaxPool2d",
        hidden_size=128,
        temppool='attention',
    )
    if pretrained:
        state = torch.load('crnn.pth')
        model.load_state_dict(state, strict=False)
    return model


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class MaxPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.max(decision, dim=self.pooldim)[0]


class LinearSoftPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return (decision**2).sum(self.pooldim) / decision.sum(self.pooldim)


class MeanPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.mean(decision, dim=self.pooldim)


class SoftPool(nn.Module):
    """docstring for SoftPool"""
    def __init__(self, T=1, pooldim=1):
        super().__init__()
        self.pooldim = pooldim
        self.T = T

    def forward(self, logits, decision):
        w = torch.softmax(decision / self.T, dim=self.pooldim)
        return torch.sum(decision * w, dim=self.pooldim)


class AutoPool(nn.Module):
    """docstring for AutoPool"""
    def __init__(self, outputdim=10, pooldim=1):
        super().__init__()
        self.outputdim = outputdim
        self.alpha = nn.Parameter(torch.ones(outputdim))
        self.dim = pooldim

    def forward(self, logits, decision):
        scaled = self.alpha * decision  # \alpha * P(Y|x) in the paper
        w = torch.softmax(scaled, dim=self.dim)
        return torch.sum(decision * w, dim=self.dim)  # Return B x 1 x C


class AttentionPool(nn.Module):
    """docstring for AttentionPool"""
    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):
        super().__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.pooldim = pooldim
        self.transform = nn.Linear(inputdim, outputdim)
        self.activ = nn.Softmax(dim=pooldim)
        self.eps = 1e-7

    def forward(self, logits, decision):
        # Input is (B, T, D)
        # B, T , D
        w = self.activ(self.transform(logits))
        # B, T, D
        detect = (decision * w).sum(self.pooldim) / w.sum(self.pooldim)
        return detect


def parse_poolingfunction(poolingfunction_name='mean', **kwargs):
    if poolingfunction_name == 'mean':
        return MeanPool(pooldim=1)
    elif poolingfunction_name == 'max':
        return MaxPool(pooldim=1)
    elif poolingfunction_name == 'linear':
        return LinearSoftPool(pooldim=1)

    elif poolingfunction_name == 'soft':
        return SoftPool(pooldim=1)
    elif poolingfunction_name == 'auto':
        return AutoPool(outputdim=kwargs['outputdim'])
    elif poolingfunction_name == 'attention':
        return AttentionPool(inputdim=kwargs['inputdim'],
                             outputdim=kwargs['outputdim'])


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)


class BiGRU(nn.Module):
    """BiGRU"""
    def __init__(self, inputdim, outputdim, bidirectional=True, **kwargs):
        nn.Module.__init__(self)

        self.rnn = nn.GRU(inputdim,
                          outputdim,
                          bidirectional=bidirectional,
                          batch_first=True,
                          **kwargs)

    def forward(self, x, hid=None):
        x, hid = self.rnn(x)
        return x, (hid, )


class ConvConvBlock(nn.Module):
    """docstring for StandardBlock"""
    def __init__(self,
                 inputfilter,
                 outputfilter,
                 kernel_size,
                 stride,
                 padding,
                 bn=True,
                 **kwargs):
        super(ConvConvBlock, self).__init__()
        self.block = nn.Sequential(
            StandardBlock(inputfilter,
                          outputfilter,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          bias=not bn),
            StandardBlock(outputfilter,
                          outputfilter,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          bias=not bn),
        )

    def forward(self, x):
        return self.block(x)


class StandardBlock(nn.Module):
    """docstring for StandardBlock"""
    def __init__(self,
                 inputfilter,
                 outputfilter,
                 kernel_size,
                 stride,
                 padding,
                 bn=True,
                 **kwargs):
        super(StandardBlock, self).__init__()
        self.activation = kwargs.get('activation', nn.ReLU())
        self.batchnorm = nn.Sequential() if not bn else nn.BatchNorm2d(
            inputfilter)
        if self.activation.__class__.__name__ == 'GLU':
            outputfilter = outputfilter * 2
        self.conv = nn.Conv2d(inputfilter,
                              outputfilter,
                              kernel_size=kernel_size,
                              stride=stride,
                              bias=not bn,
                              padding=padding)

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.conv(x)
        return self.activation(x)


class CRNN(nn.Module):
    """Encodes the given input into a fixed sized dimension"""
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self._inputdim = inputdim
        self._filtersizes = kwargs.get('filtersizes', [3, 3, 3, 3, 3])
        self._filter = kwargs.get('filter', [16, 32, 128, 128, 128])
        self._pooling = kwargs.get('pooling', [2, 2, (1, 2), (1, 2)])
        self._hidden_size = kwargs.get('hidden_size', 128)
        self._bidirectional = kwargs.get('bidirectional', True)
        self._rnn = kwargs.get('rnn', 'BiGRU')
        self._pooltype = kwargs.get('pooltype', 'MaxPool2d')
        self._activation = kwargs.get('activation', 'ReLU')
        self._blocktype = kwargs.get('blocktype', 'StandardBlock')
        if not isinstance(self._blocktype, list):
            self._blocktype = [self._blocktype] * len(self._filtersizes)
        self._bn = kwargs.get('bn', True)
        activation_kwargs = {}
        if self._activation == 'GLU':
            activation_kwargs = {'dim': 1}
        poolingtypekwargs = {}
        if self._pooltype == 'LPPool2d':
            poolingtypekwargs = {"norm_type": 2}

        self._filter = [1] + self._filter
        net = nn.ModuleList()
        assert len(self._filter) - 1 == len(self._filtersizes)
        for nl, (h0, h1, filtersize, poolingsize, blocktype) in enumerate(
                zip_longest(self._filter, self._filter[1:], self._filtersizes,
                            self._pooling, self._blocktype)):
            # Stop in zip_longest when last element arrived
            if not h1:
                break
            # For each layer needs to know the filter size
            if self._pooltype in ('ConvolutionPool', 'GatedPooling'):
                poolingtypekwargs = {'filter': h1}
            current_activation = getattr(nn,
                                         self._activation)(**activation_kwargs)
            net.append(globals()[blocktype](inputfilter=h0,
                                            outputfilter=h1,
                                            kernel_size=filtersize,
                                            padding=int(filtersize) // 2,
                                            bn=self._bn,
                                            stride=1,
                                            activation=current_activation))
            # Poolingsize will be None if pooling is finished
            if poolingsize:
                net.append(
                    getattr(nn, self._pooltype)(kernel_size=poolingsize,
                                                **poolingtypekwargs))
            # Only dropout at last layer before GRU
            if nl == (len(self._filter) - 2):
                net.append(nn.Dropout(0.3))
        self.network = nn.Sequential(*net)

        def calculate_cnn_size(input_size):
            x = torch.randn(input_size).unsqueeze(0)
            output = self.network(x)
            return output.size()[1:]

        cnn_outputdim = calculate_cnn_size((1, 500, inputdim))
        self.rnn = globals()[self._rnn](self._filter[-1] * cnn_outputdim[-1],
                                        self._hidden_size, self._bidirectional)
        rnn_output = self.rnn(
            torch.randn(1, 500,
                        self._filter[-1] * cnn_outputdim[-1]))[0].shape[-1]
        # During training, pooling in time
        self.outputlayer = nn.Linear(rnn_output, outputdim)
        self.temp_pool = parse_poolingfunction(kwargs.get(
            'temppool', 'linear'),
                                               inputdim=rnn_output,
                                               outputdim=outputdim)
        self.network.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x):
        # Add dimension for filters
        x = x.unsqueeze(1)
        x = self.network(x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        x, _ = self.rnn(x)
        decision_time = torch.sigmoid(self.outputlayer(x))
        decision = self.temp_pool(x, decision_time).squeeze(1)
        decision = torch.clamp(decision, min=1e-7, max=1.)
        return decision, decision_time


class ConvBNReLU(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False), nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim,
                       hidden_dim,
                       stride=stride,
                       groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class MobileCRNN(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        filters = [1] + kwargs.get('filters', [40] + [160] * 5)
        kernels = kwargs.get('kernels', [5] + [3] * 5)
        paddings = kwargs.get('padding', [2] + [1] * 5)
        strides = kwargs.get('strides', [2] + [1] * 5)
        poolings = kwargs.get('pooling',
                              [(2, 4)] + [(1, 2)] * 3 + [(1, 1)] * 2)
        features = nn.ModuleList()
        for h0, h1, kernel_size, padding, pooling, stride in zip(
                filters, filters[1:], kernels, paddings, poolings, strides):
            if h0 == 1:
                features.append(
                    nn.Sequential(
                        nn.BatchNorm2d(h0),
                        nn.Conv2d(
                            h0,
                            h1,
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride,
                        ), Swish()))
            else:
                features.append(InvertedResidual(h0, h1, 1, expand_ratio=6))
            if np.prod(pooling) > 1:  # just dont append if its (1,1)
                features.append(nn.MaxPool2d(pooling))
        self.features = nn.Sequential(*features)
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,
                                                      inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = BiGRU(inputdim=rnn_input_dim, outputdim=128)
        self.temp_pool = parse_poolingfunction(kwargs.get(
            'temppool', 'linear'),
                                               inputdim=int(256),
                                               outputdim=outputdim)
        self.outputlayer = nn.Linear(256, outputdim)

    def forward(self, x, mode='all'):
        if mode == 'all':
            x = x.unsqueeze(1)
            x = self.features(x)
            x = x.transpose(1, 2).contiguous().flatten(-2)
            x, _ = self.gru(x)
            decision_time = torch.sigmoid(self.outputlayer(x))
            decision_time = torch.clamp(decision_time, min=1e-7, max=1.)
            decision = self.temp_pool(x, decision_time).squeeze(1)
            decision = torch.clamp(decision, min=1e-7, max=1.)
            return decision, decision_time
            
        if mode == 'embed':
            x = x.unsqueeze(1)
            x = self.features(x)
            return x
        
        elif mode == 'cont':
            x = x.transpose(1, 2).contiguous().flatten(-2)
            x, _ = self.gru(x)
            decision_time = torch.sigmoid(self.outputlayer(x))
            decision_time = torch.clamp(decision_time, min=1e-7, max=1.)
            decision = self.temp_pool(x, decision_time).squeeze(1)
            decision = torch.clamp(decision, min=1e-7, max=1.)
            return decision, decision_time


if __name__ == "__main__":
    model = mobilecrnn_v2()
    print(model)
