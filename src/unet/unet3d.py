'''
From https://github.com/wolny/pytorch-3dunet.
'''
import torch.nn as nn
import torch
from functools import partial
from torch.nn import functional as F
import lightning as L
from torch import optim
from ..constants import atom_cls,atom_cls_weight, ch_mass_cls, ch_mass_cls_weight, ch_cls,cls_ch_weight


def conv3d(in_channels, out_channels, kernel_size, bias, padding):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple):
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups,
                                   padding=padding))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups,
                                   padding=padding))


class ExtResNetBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf

    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, **kwargs):
        super(ExtResNetBlock, self).__init__()

        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order,
                                num_groups=num_groups)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out

        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=2, pool_type='max', basic_module=DoubleConv, conv_layer_order='gcr',
                 num_groups=8, padding=1):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer
    (either learned ConvTranspose3d or nearest neighbor interpolation) followed by a basic module (DoubleConv or ExtResNetBlock).
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upsample (boole): should the input be upsampled
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=(2, 2, 2), basic_module=DoubleConv,
                 conv_layer_order='gcr', num_groups=8, mode='nearest', padding=1, upsample=True):
        super(Decoder, self).__init__()

        if upsample:
            if basic_module == DoubleConv:
                # if DoubleConv is the basic_module use interpolation for upsampling and concatenation joining
                self.upsampling = InterpolateUpsampling(mode=mode)
                # concat joining
                self.joining = partial(self._joining, concat=True)
            else:
                # if basic_module=ExtResNetBlock use transposed convolution upsampling and summation joining
                self.upsampling = TransposeConvUpsampling(in_channels=in_channels, out_channels=out_channels,
                                                          kernel_size=conv_kernel_size, scale_factor=scale_factor)
                # sum joining
                self.joining = partial(self._joining, concat=False)
                # adapt the number of in_channels for the ExtResNetBlock
                in_channels = out_channels
        else:
            # no upsampling
            self.upsampling = NoUpsampling()
            # concat joining
            self.joining = partial(self._joining, concat=True)

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)
        

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


def create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                    pool_kernel_size):
    # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            encoder = Encoder(in_channels, out_feature_num,
                              apply_pooling=False,  # skip pooling in the firs encoder
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding)
        else:
            # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
            encoder = Encoder(f_maps[i - 1], out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              pool_kernel_size=pool_kernel_size,
                              padding=conv_padding)

        encoders.append(encoder)

    return nn.ModuleList(encoders)


def create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, upsample):
    # create decoder path consisting of the Decoder modules. The length of the decoder list is equal to `len(f_maps) - 1`
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        if basic_module == DoubleConv:
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        else:
            in_feature_num = reversed_f_maps[i]

        out_feature_num = reversed_f_maps[i + 1]

        # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
        # currently strides with a constant stride: (2, 2, 2)

        _upsample = True
        if i == 0:
            # upsampling can be skipped only for the 1st decoder, afterwards it should always be present
            _upsample = upsample

        decoder = Decoder(in_feature_num, out_feature_num,
                          basic_module=basic_module,
                          conv_layer_order=layer_order,
                          conv_kernel_size=conv_kernel_size,
                          num_groups=num_groups,
                          padding=conv_padding,
                          upsample=_upsample)
        decoders.append(decoder)
    return nn.ModuleList(decoders)


class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode='nearest'):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class TransposeConvUpsampling(AbstractUpsampling):
    """
    Args:
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True

    """

    def __init__(self, in_channels=None, out_channels=None, kernel_size=3, scale_factor=(2, 2, 2)):
        # make sure that the output size reverses the MaxPool3d from the corresponding encoder
        upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor,
                                      padding=1)
        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        return x

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]

class ResUNet3D4EM(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        testing (bool): if True (testing mode) the `final_activation` (if present, i.e. `is_segmentation=true`)
            will be applied as the last operation during the forward pass; if False the model is in training mode
            and the `final_activation` (even if present) won't be applied; default: False
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, basic_module=ExtResNetBlock, task='mrc2sim', in_channels=1, out_channels=21, f_maps=32, layer_order='gcr',
                 num_groups=8, num_levels=5, conv_kernel_size=3, pool_kernel_size=2, conv_padding=1, finetune=False, **kwargs):
        super(ResUNet3D4EM, self).__init__()
        self.task=task

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        self.AA_encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,num_groups, pool_kernel_size)
        self.AA_decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,upsample=True)


        self.AA_final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        self.final_activation=nn.Softmax(dim=1)
        self.finetune=finetune

        self.seq_linear = nn.Linear(22, 256)
        self.conv_linear = nn.Linear(256,22)


    def forward(self, x, seq=None, mask=None):
        if self.finetune:
            with torch.no_grad():
                encoders_features = []
                for encoder in self.AA_encoders:
                    x = encoder(x)
                    encoders_features.insert(0, x)
                encoders_features = encoders_features[1:]
                for decoder, encoder_features in zip(self.AA_decoders, encoders_features):
                    x = decoder(encoder_features, x)
        else:
            encoders_features = []
            for encoder in self.AA_encoders:
                x = encoder(x)
                encoders_features.insert(0, x)
            encoders_features = encoders_features[1:]
            for decoder, encoder_features in zip(self.AA_decoders, encoders_features):
                x = decoder(encoder_features, x)

        AA_output = self.AA_final_conv(x)
        return AA_output
    
class UNet3d(L.LightningModule):
    def __init__(self,lr=1e-4,precision='32',label_smoothing=0.5,num_levels=5,f_maps=32,loss_reduction='mean',task='ring_cls',more_aa=True,finetune=False,loss_weight=False):
        print(lr,precision,label_smoothing,num_levels,f_maps,loss_reduction,task)
        super().__init__()
        if task=='atom_cls':
            model=ResUNet3D4EM(out_channels=2,num_levels=num_levels,f_maps=f_maps,finetune=finetune)
        elif task=='ring_cls':
            model=ResUNet3D4EM(out_channels=4,num_levels=num_levels,f_maps=f_maps,finetune=finetune)
        elif task=='find_ch':
            model=ResUNet3D4EM(out_channels=3,num_levels=num_levels,f_maps=f_maps,finetune=finetune)
        elif task=='cls_ch':
            model=ResUNet3D4EM(out_channels=3,num_levels=num_levels,f_maps=f_maps,finetune=finetune)
        self.emmodel= model
        self.task=task
        self.lr=lr
        self.label_smoothing=label_smoothing
        self.precision=precision
        self.more_aa=more_aa
        self.loss_reduction=loss_reduction
        self.loss_weight=loss_weight
        self.loss_rec=torch.tensor(0.0,requires_grad=True)

    def step_mrc_to_eman(self,batch):
        mrcs,emans,eman_masks=batch
        outputs=self.emmodel(mrcs).float()
        # loss = nn.functional.mse_loss(outputs, emans.float())
        # return loss
        outputs=torch.masked_select(outputs,eman_masks).float()
        if outputs.shape[0]>0:
            emans=torch.masked_select(emans,eman_masks).float()
            loss = nn.functional.mse_loss(outputs, emans).float()
            return loss
        else:
            return torch.tensor(0,requires_grad=True).cuda().float()
        
    

    def on_train_epoch_start(self):
        self.train_match_num=0
        self.train_match_num2=0
        self.train_rec_num=0
        self.train_pred_num=0
        
    
    def on_train_epoch_end(self):
        self.train_recall=self.train_match_num/self.train_rec_num if self.train_rec_num!=0 else 0
        self.train_precision=self.train_match_num2/self.train_pred_num if self.train_pred_num!=0 else 0
        self.log("train_recall", self.train_recall)
        self.log("train_precision", self.train_precision)

    def on_validation_epoch_start(self):
        self.loss_rec=torch.tensor(0.0,requires_grad=True)
        self.val_match_num=0
        self.val_match_num2=0
        self.val_rec_num=0
        self.val_pre_num=0

    def on_validation_epoch_end(self):
        self.val_recall=self.val_match_num/self.val_rec_num if self.val_rec_num!=0 else 0
        self.val_precision=self.val_match_num2/self.val_pre_num if self.val_pre_num!=0 else 0
        self.log("val_recall", self.val_recall)
        self.log("val_precision", self.val_precision)

    def on_test_epoch_start(self):
        self.loss_rec=torch.tensor(0.0,requires_grad=True)
        self.test_acc_num=0
        self.test_all_num=0

    def on_test_epoch_end(self):
        assert self.test_all_num!=0
        self.test_acc=self.test_acc_num/self.test_all_num
        self.log("test_acc", self.test_acc)
        
    def step_aa(self,batch):
        if self.more_aa:
            # mrcs,aas,aa_diffs,aa_weights,seqs,seq_masks=batch
            mrcs,aas,aa_diffs,aa_weights=batch
            aa_diffs=aa_diffs
        else:
            mrcs,aas=batch

        mask=(aas!=0)*(aas!=21)
        if mask.sum()!=0:
            # outputs=self.emmodel(mrcs,seqs,seq_masks)
            outputs=self.emmodel(mrcs)
            outputs=outputs.float()
            if self.more_aa:
                loss = nn.functional.cross_entropy(outputs, aa_diffs.long(),ignore_index=0,label_smoothing=self.label_smoothing,reduction=self.loss_reduction)
            else:
                loss = nn.functional.cross_entropy(outputs, aas,ignore_index=0,label_smoothing=self.label_smoothing,reduction=self.loss_reduction)
            if self.loss_weight:
                print(loss.shape)
                print(aa_weights.shape)
                loss=(loss*aa_weights).mean()
            result=torch.argmax(nn.Softmax(dim=1)(outputs[:,1:-1,4:60,4:60,4:60]), 1)
            mask_clip=mask[:,4:60,4:60,4:60]
            acc_num=(result[mask_clip]==(aas[:,4:60,4:60,4:60]-1)[mask_clip]).sum()
            return loss,acc_num, mask_clip.sum()
        else:
            loss=torch.tensor(0.0,requires_grad=True)
            return loss,0,0
    def step_ring_cls(self,batch):
        mrcs,labels=batch
        outputs=self.emmodel(mrcs)
        cls_weight = torch.tensor(atom_cls_weight, device=outputs.device)
        loss=nn.functional.cross_entropy(outputs, labels,weight=cls_weight,ignore_index=2,label_smoothing=self.label_smoothing,reduction=self.loss_reduction)
        if torch.sum(labels==atom_cls['in_ring_4_7'])>0:
            pred_ring=torch.argmax(nn.Softmax(dim=1)(outputs), 1)
            match_num=torch.sum((labels==atom_cls['in_ring_4_7'])*(labels==pred_ring))
            match_num2=torch.sum((pred_ring==atom_cls['in_ring_4_7'])*(labels>=atom_cls['in_ring_4_7_box333']))
            rec_num=torch.sum(labels==atom_cls['in_ring_4_7'])
            pre_num=torch.sum(pred_ring==atom_cls['in_ring_4_7'])
            return loss, match_num,match_num2,rec_num,pre_num
        else:
            return loss,0,0,0,0
    
    def step_find_ch(self,batch):
        mrcs,labels=batch
        outputs=self.emmodel(mrcs)
        cls_weight = torch.tensor(ch_mass_cls_weight, device=outputs.device)
        loss=nn.functional.cross_entropy(outputs, labels,weight=cls_weight,ignore_index=0,label_smoothing=self.label_smoothing,reduction=self.loss_reduction)
        
        pred_ch=torch.argmax(nn.Softmax(dim=1)(outputs), 1)
        match_num=torch.sum((pred_ch==ch_mass_cls['ch_mass_center'])*(labels==ch_mass_cls['ch_mass_center']))
        rec_num=torch.sum(labels==ch_mass_cls['ch_mass_center'])
        pre_num=torch.sum(pred_ch==ch_mass_cls['ch_mass_center'])
        
        return loss, match_num,rec_num,pre_num

    def step_cls_ch(self,batch):
        mrcs,labels=batch
        outputs=self.emmodel(mrcs)
        cls_weight = torch.tensor(cls_ch_weight, device=outputs.device)
        loss=nn.functional.cross_entropy(outputs, labels,weight=cls_weight,ignore_index=0,label_smoothing=self.label_smoothing,reduction=self.loss_reduction)
        
        pred_ch=torch.argmax(nn.Softmax(dim=1)(outputs), 1)
        match_num=torch.sum((pred_ch==ch_cls['in_furanose'])*(labels==ch_cls['in_furanose']))
        rec_num=torch.sum(labels==ch_cls['in_furanose'])

        match_num2=torch.sum(((pred_ch==ch_cls['in_furanose'])*(labels==ch_cls['in_furanose']))[labels>=ch_cls['in_furanose']])
        pre_num=torch.sum(pred_ch[labels>=ch_cls['in_furanose']]==ch_cls['in_furanose'])
        
        return loss, match_num, match_num2, rec_num,pre_num

    def training_step(self, batch, batch_idx):
        # outputs=torch.masked_select(self.emmodel(mrcs)[:,0],masks).float()
        # log_ca_nums=torch.masked_select(log_ca_nums,masks).float()
        
        if self.task=='ring_cls':
            loss,match_num,match_num2,rec_num,pre_num =self.step_ring_cls(batch)
            if loss.item()!=0:
                self.loss_rec=loss.item()
                self.log("train_loss", loss)
            else:
                self.log("train_loss", self.loss_rec)
            self.train_match_num+=match_num
            self.train_match_num2+=match_num2
            self.train_rec_num+=rec_num
            self.train_pred_num+=pre_num
            # self.log("train_acc_num", acc_num)
            return loss
        elif self.task=='find_ch':
            loss,match_num,rec_num,pre_num =self.step_find_ch(batch)
            if loss.item()!=0:
                self.loss_rec=loss.item()
                self.log("train_loss", loss)
            else:
                self.log("train_loss", self.loss_rec)
            self.train_match_num+=match_num
            self.train_match_num2+=match_num
            self.train_rec_num+=rec_num
            self.train_pred_num+=pre_num
            # self.log("train_acc_num", acc_num)
            return loss
        elif self.task=='cls_ch':
            loss,match_num,match_num2,rec_num,pre_num =self.step_cls_ch(batch)
            if loss.item()!=0:
                self.loss_rec=loss.item()
                self.log("train_loss", loss)
            else:
                self.log("train_loss", self.loss_rec)
            self.train_match_num+=match_num
            self.train_match_num2+=match_num2
            self.train_rec_num+=rec_num
            self.train_pred_num+=pre_num
            # self.log("train_acc_num", acc_num)
            return loss

    def validation_step(self, batch, batch_idx):
        if self.task=='ring_cls':
            loss,match_num,match_num2,rec_num,pre_num =self.step_ring_cls(batch)
            if loss.item()!=0:
                self.loss_rec=loss.item()
                self.log("val_loss", loss)
            else:
                self.log("val_loss", self.loss_rec)
            self.val_match_num+=match_num
            self.val_match_num2+=match_num2
            self.val_rec_num+=rec_num
            self.val_pre_num+=pre_num
            return loss
        elif self.task=='find_ch':
            loss,match_num,rec_num,pre_num =self.step_find_ch(batch)
            if loss.item()!=0:
                self.loss_rec=loss.item()
                self.log("val_loss", loss)
            else:
                self.log("val_loss", self.loss_rec)
            self.val_match_num+=match_num
            self.val_match_num2+=match_num
            self.val_rec_num+=rec_num
            self.val_pre_num+=pre_num
            # self.log("train_acc_num", acc_num)
            return loss
        elif self.task=='cls_ch':
            loss,match_num,match_num2,rec_num,pre_num =self.step_cls_ch(batch)
            if loss.item()!=0:
                self.loss_rec=loss.item()
                self.log("val_loss", loss)
            else:
                self.log("val_loss", self.loss_rec)
            self.val_match_num+=match_num
            self.val_match_num2+=match_num2
            self.val_rec_num+=rec_num
            self.val_pre_num+=pre_num
            # self.log("train_acc_num", acc_num)
            return loss
        
    # def test_step(self, batch, batch_idx):
    #     if self.task=='mrc2sim':
    #         loss=self.step_mrc_to_eman(batch)
    #         self.log("test_loss", loss)
    #         return loss
    #     elif self.task=='aa_cls':
    #         loss,acc_num, all_num=self.step_aa(batch)
    #         if loss.item()!=0:
    #             self.loss_rec=loss.item()
    #             self.log("test_loss", loss)
    #         else:
    #             self.log("test_loss", self.loss_rec)
    #         self.val_acc_num+=acc_num
    #         self.val_all_num+=all_num
    #         # self.log("val_acc", acc)
    #         return loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    