# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import torch.nn.functional as F
from deformable_attention import DeformableAttention1D



#  ========================================================== RESNET50_3D IMPLEMENTATION =================================================



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block = Bottleneck,
            layers = [3, 4, 6, 3],
            zero_init_residual=False,
            groups=1,
            widen=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            return_origin = False, 
            normalize=False,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.return_origin = return_origin
        self.padding = nn.ConstantPad2d(1, 0.0)

        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # change padding 3 -> 2 compared to original torchvision code because added a padding layer
        num_out_filters = width_per_group * widen
        self.conv1 = nn.Conv2d(
            3, num_out_filters, kernel_size=7, stride=2, padding=2, bias=False
        )
        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(
            block, num_out_filters, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        num_out_filters *= 2
        self.layer3 = self._make_layer(
            block, num_out_filters, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        num_out_filters *= 2
        self.layer4 = self._make_layer(
            block, num_out_filters, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.deconv = nn.ConvTranspose1d(in_channels = 5888, out_channels = 640,kernel_size = 1, stride = 1, padding = 0)
        # normalize output features
        self.l2norm = normalize

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def flatten_feature(self, x) :
        x = torch.flatten(self.avgpool(x), start_dim = 1)
        return x
    
    def forward_2D(self, x):
        x = self.padding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x1 = self.flatten_feature(x)
#         print(x1.shape)
        x1 = nn.functional.normalize(x1, dim=1, p=2)
        x = self.layer2(x)
        x2 = self.flatten_feature(x)
#         print(x2.shape)
        x2 = nn.functional.normalize(x2, dim=1, p=2)
        x = self.layer3(x)
        x3 = self.flatten_feature(x)
        x3 = nn.functional.normalize(x3, dim=1, p=2)
#         print(x3.shape)
        x = self.layer4(x)
        x4 = self.flatten_feature(x)
        x4 = nn.functional.normalize(x4, dim=1, p=2)
#         print(x4.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
#         print(x.shape)
        x = nn.functional.normalize(x, dim=1, p=2)
        out = torch.cat( [x1,x2,x3,x4,x]  ,  1) # [1, n_frames, emb_dim]
        return out
        
    
    
    def forward(self, batch_3D) :
        B, N, C, W, H = batch_3D.shape
        x = rearrange(batch_3D , 'B N C W H -> (B N) C W H')
        res_out = self.forward_2D(x)
        res_out = rearrange(res_out, '(B N) D -> B N D', B = B , N = N ) # [batch_size, n_frames, emb_dim']
        if self.return_origin :
            return res_out
        res_out = rearrange(res_out, 'B N D -> B D N')
        res_out = self.deconv(res_out)                                     # [batch_size, n_frames, emb_dim]
        res_out = rearrange(res_out, 'B D N -> B N D')
        if self.l2norm:
            res_out = nn.functional.normalize(res_out, dim=2, p=2)
        
        return res_out # [batch_size, n_frames, emb_dim]


class ProjectionHead(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim, normalize = False ) :
        super(ProjectionHead, self).__init__()
        self.normalize = normalize
        self.projection_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self,x) :

        x = self.projection_head(x)
        if self.normalize:
            x = nn.functional.normalize(x, dim=1, p=2)
        return x
    
class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out


#  ========================================================== TRANSFORMER IMPLEMENTATION =================================================

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.activation = self._get_activation_fn("gelu")
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
    def _get_activation_fn(self,activation):
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

    

class EncoderLayer(nn.Module):

    def __init__(self,n_frames,d_model, ffn_hidden, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention =  DeformableAttention1D(
                                dim = n_frames,
                                downsample_factor = 4,
                                offset_scale = 2,
                                offset_kernel_size = 6
                            )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        
    def compute_attention(self,src) :
        return torch.stack([self.attention(src[i]) for i in range(batch_size)])

    def forward(self, x):
        # 1. compute self attention
        _x = x
        x = self.attention(x)
        # 2. add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        # 4. add and norm
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x
    
    
    
class EncoderLayer_v2(nn.Module):

    def __init__(self,n_frames,d_model, ffn_hidden, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention =  DeformableAttention1D(
                                dim = n_frames,
                                downsample_factor = 4,
                                offset_scale = 2,
                                offset_kernel_size = 6
                            )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        
    
    def compute_attention(self,src) :
        return torch.stack([self.attention(src[i]) for i in range(batch_size)])

    def forward(self, x):
        # 1. compute self attention
        _x = x
        x = self.attention(x)
        # 2. add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        # 4. add and norm
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x

class DeTrans(nn.Module):
    def __init__(self,n_frames, d_model, ffn_hidden, drop_prob, n_layers):
        super(DeTrans,self).__init__()
        self.layers = nn.ModuleList([ EncoderLayer(
                                                   n_frames = n_frames,
                                                   d_model = d_model, 
                                                   ffn_hidden = ffn_hidden,
                                                   drop_prob = drop_prob)
                                     for _ in range(n_layers)])
    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x
    
    
class EncoderModel(nn.Module):
    def __init__(self, 
                 n_frames : int, 
                 d_model : int, 
                 ffn_hidden : int,
                 drop_prob : float,
                 n_layers : int):
        
        super(EncoderModel,self).__init__()
        self.encoder = DeTrans(
                                n_frames = n_frames, 
                                d_model = d_model, 
                                ffn_hidden = ffn_hidden,
                                drop_prob = drop_prob, 
                                n_layers = n_layers)


    def forward(self, src: Tensor) -> Tensor:
        return self.encoder(src) 
    

class EncoderDecoderModel(nn.Module):
    def __init__(self, 
                 n_frames : int, 
                 d_model : int, 
                 ffn_hidden : int,
                 drop_prob : float,
                 n_layers : int,
                 mask_ratio : float):
        
        super().__init__()
        self.model_type = 'Transformer'
        self.encoder = EncoderModel(
                 n_frames = n_frames, 
                 d_model = d_model, 
                 ffn_hidden = ffn_hidden,
                 drop_prob = drop_prob,
                 n_layers = n_layers
        )
        
        self.mask_ratio = mask_ratio
        self.n_frames = n_frames
        self.mask_token = nn.Parameter(torch.randn(d_model))
        self.n_masked_tokens = int(mask_ratio  *   n_frames)
        self.decoder = nn.Linear(d_model, d_model)
            
    def forward(self, src, mask):
        
        batch_size, n_frames, d_model = src.shape
        mask_tokens = self.mask_token.repeat(batch_size, n_frames, 1)
        mask_embed = (~mask)*src + mask*mask_tokens
        output = self.decoder(self.encoder(mask_embed)) 
        return output
        
    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        

class ResNet_Trans(nn.Module):
    def __init__(self, intra, inter, 
                 prototypes = None
                ) :
        super(ResNet_Trans, self).__init__()
        self.intra = intra
        self.inter = inter
        self.prototypes = prototypes

    def forward(self,x) :

        x = self.intra(x)
        x = self.inter(x)
        x = torch.mean(x,1)
        x = nn.functional.normalize(x, dim=1, p=2)
        if self.prototypes is not None:
            return x, self.prototypes(x)
        
        return x 
    

    
    
def saveModel(model, epoch, optimizer, best_acc):
    path = "/home/hnguyen/Research/Downstream_Tasks/LUNA_Classification/Weight/Log/model_308/epoch_ " + str(
        epoch) + ".pth"

    save_dicts = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_acc": best_acc,
    }
    torch.save(save_dicts, path)

# def resnet50(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


# def resnet50w2(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], widen=2, **kwargs)


# def resnet50w4(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], widen=4, **kwargs)


# def resnet50w5(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], widen=5, **kwargs)

