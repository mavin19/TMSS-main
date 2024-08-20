import torch
from torch import nn
from typing import Tuple, Union
from src.models.unetrpp.dynunet_block import UnetOutBlock, UnetResBlock
from src.models.unetrpp.model_components import UnetrPPEncoder, UnetrUpBlock
from torchmtlr.torchmtlr import MTLR
import sys
from monai.utils import optional_import
from src.models.unetrpp.resnet import ResNet18_3D, ResNet50_3D
import hydra
einops, _ = optional_import("einops")

@hydra.main(config_path="configs/", config_name="train.yaml")

def flatten_layers(arr):
    return [i for sub in arr for i in sub]


class UNETR_PP(nn.Module):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            hparams: dict,
            in_channels: int,
            out_channels: int,
            feature_size: int = 16,
            hidden_size: int = 256,
            num_heads: int = 8,
            pos_embed: str = "perceptron",
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.25,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimensions of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        # 这段代码首先调用父类的初始化方法，然后初始化一些参数。如果depths未提供，则将其默认设置为[3, 3, 3, 3]。
        # dropout_rate应该在0和1之间，如果不在这个范围内，则会抛出异常。
        # 检查 pos_embed是否是有效的嵌入层类型。

        # This code first calls the parent class's initialization method and then initializes some parameters.
        # If depths is not provided, it is set to [3, 3, 3, 3] by default.
        # dropout_rate should bebetween 0 and 1, and if it is not in this range, an exception will be thrown.
        # Check whether pos_embed is a valid embed layer type.
        self.feat_size = (2, 5, 5,)         # baseline   c h w
        self.hidden_size = hidden_size
        # 定义特征尺寸和隐藏层尺寸。
        # Define the feature size and the hidden layer size.

        self.unetr_pp_encoder = UnetrPPEncoder(dims=dims, depths=depths, num_heads=4)
        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=4 * 10 * 10,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=8 * 20 * 20,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16 * 40 * 40,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(4, 4, 4),  # (1,4,4)
            norm_name=norm_name,
            out_size=64 * 160 * 160,
            conv_decoder=True,
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

        # 初始化了UnetrPPEncoder编码器和多个UnetResBlock和UnetrUpBlock解码器模块。

        # The UnetrPPEncoder encoder and several UnetResBlock and UnetrUpBlock decoder modules are initialized.

        # ------------------------   survival ----------------

        if hparams['n_dense'] <= 0:     # fc_layers
            self.mtlr = MTLR(hparams['hidden_size'], hparams['time_bins'])

        else:
            fc_layers = [[nn.Linear(128, 64 * hparams['dense_factor']),  # * 2 means cat
                          nn.BatchNorm1d(64 * hparams['dense_factor']),
                          nn.ReLU(inplace=True),
                          nn.Dropout(hparams['dropout'])]]

            if hparams['n_dense'] > 1:
                fc_layers.extend(
                    [[nn.Linear(64 * hparams['dense_factor'], 64),  # 4 * hparams['dense_factor']
                      nn.BatchNorm1d(64),
                      nn.ReLU(inplace=True),
                      nn.Dropout(hparams['dropout'])], ])
                for _ in range(hparams['n_dense'] - 2):
                    fc_layers.append(
                        [nn.Linear(64, 64),  # 4 * hparams['dense_factor']
                         nn.BatchNorm1d(64),
                         nn.ReLU(inplace=True),
                         nn.Dropout(hparams['dropout'])])

            fc_layers = flatten_layers(fc_layers)

            self.mtlr_fc = nn.Sequential(*fc_layers)    # before input
            self.mtlr = MTLR(64, hparams['time_bins'])
        # 根据hparams['n_dense']参数，定义MTLR层或者多层感知机（MLP）层。
        # The MTLR layer or multi - layer perceptron(MLP) layer is defined according to the hparams['n_dense'] parameter.
        self.EHR_proj_encoder = nn.Sequential(nn.Linear(6, 160 * 160))
        # self.img_proj = nn.Sequential(nn.Linear(1638400, 64), nn.BatchNorm1d(64), nn.Dropout(0.25),
        #                              # nn.Linear(64, 128), nn.BatchNorm1d(128), nn.Dropout(0.25),
        #                               nn.Linear(64, 64), nn.BatchNorm1d(64), nn.Dropout(0.25))
        self.img_proj = nn.Sequential(nn.Linear(1638400, 128), nn.LeakyReLU(), nn.BatchNorm1d(128),  # 128
                                      nn.Linear(128, 64), nn.LeakyReLU(), nn.BatchNorm1d(64),
                                      )
        # self.EHR_proj = nn.Sequential(nn.Linear(38, 32), nn.LeakyReLU(), nn.BatchNorm1d(32),  # 32
        #                               nn.Linear(32, 64), nn.LeakyReLU(), nn.BatchNorm1d(64),
        #                               nn.Linear(64, 64), nn.LeakyReLU(), )

        self.EHR_proj = nn.Sequential(nn.Linear(38, 32), nn.LeakyReLU(), nn.BatchNorm1d(32),
                                      nn.Linear(32, 64), nn.LeakyReLU(), nn.BatchNorm1d(64),)

        # Define the projection layer of clinical text and the projection layer of image features.
        # if you do not want to use Batch Normalization, delete the nn.BatchNorm1d(**) of the projection layers.

        #----------dim reducetion-------#
        self.img_resnet = ResNet18_3D(num_classes=64)   # output dim = 64

        #----clip--------#
        self.text_to_vision = nn.Linear(512, 256)
        self.word_embedding = torch.load('./txt_encoding.pth')
        # add nn.Linear   64  -> 64

        #----------------------encoder------------------
        # dim = 128
        # self.conv1 = nn.Sequential(
        #     nn.Conv3d(dim // 4, dim // 2, kernel_size=7, stride=2, padding=3),
        #     nn.InstanceNorm3d(dim // 2),
        #     nn.LeakyReLU(),
        #     # nn.Upsample(scale_factor=0.5, mode="trilinear", align_corners=False),
        #
        #     nn.Conv3d(dim // 2, dim, kernel_size=7, stride=2, padding=3),
        #     nn.InstanceNorm3d(dim),
        #     nn.LeakyReLU(),
        #     # nn.Upsample(scale_factor=0.5, mode="trilinear", align_corners=False),
        # )
        #
        # self.conv2 = nn.Sequential(
        #     nn.Conv3d(dim // 2, dim, kernel_size=7, stride=2, padding=3),
        #     nn.InstanceNorm3d(dim),
        #     nn.LeakyReLU(),
        #     # nn.Upsample(scale_factor=0.5, mode="trilinear", align_corners=False),
        # )
        # self.conv4 = nn.Sequential(
        #     nn.Conv3d(dim * 2, dim, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(dim),
        #     nn.LeakyReLU(),
        #     nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        # )
        # self.conv_out = nn.Sequential(
        #     nn.Conv3d(dim * 4, dim, kernel_size=1, stride=1, padding=0),
        #     nn.InstanceNorm3d(dim),
        # )
        # self.conv_dec = nn.Sequential(
        #     nn.Conv3d(1, 1, kernel_size=17, stride=16, padding=7),
        #     # nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(1),
        #     nn.LeakyReLU(),
        #     # nn.Upsample(scale_factor=16, mode="trilinear", align_corners=False),
        # )
        # ----------------------encoder------------------

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        # 这一行代码将输入张量x的形状变为(batch_size, feat_size[0], feat_size[1], feat_size[2], hidden_size)。其中：
        # x.size(0)表示批次大小（batch size）。
        # feat_size[0], feat_size[1], feat_size[2]表示特征张量的三个维度。
        # hidden_size表示隐藏层的尺寸。

        # This line of code changes the shape of the input tensor x to (batch_size, feat_size[0], feat_size[1], feat_size[2], hidden_size). Among them:
        # x.size(0) indicates the batch size.
        # feat_size[0], feat_size[1], and feat_size[2] represent the three dimensions of the feature tensor.
        # hidden_size Indicates the size of the hidden layer.
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        # 将张量的维度从(batch_size, feat_size[0], feat_size[1], feat_size[2], hidden_size)
        # 交换为(batch_size, hidden_size, feat_size[0], feat_size[1], feat_size[2])
        # Swap the dimensions of the tensor from (batch_size, feat_size[0], feat_size[1], feat_size[2], hidden_size)
        # to(batch_size, hidden_size, feat_size[0], feat_size[1], feat_size[2])
        return x

    def forward(self, x_in):
        img, clin_var = x_in

        task_encoding = F.relu(self.text_to_vision(self.word_embedding))
        task_encoding = task_encoding.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        # 这部分代码通过将加载的文本编码(word_embedding)经过text_to_vision层，并应用ReLU激活函数，将其转化为视觉任务编码。
        # 然后通过 unsqueeze 将其扩展为四维，以适应后续的张量操作。
        # This part of the code by embedding the loaded text encoding(word_embedding) through the  text_to_vision layer,
        # and apply the ReLU activation function to convert  it into the visual task encoding.
        # It is then extended to four dimensions  by unsqueeze  to accommodate subsequent tensor operations.

        # x_in = (img['input'], clin_var)

        # print('img shape  ', img['input'].shape)       #  torch.Size([2, 1, 160, 160, 64])
        # print("clin_var.shape  ", clin_var.shape)    # B, n_clin_var

        # clin_var_encoder = self.EHR_proj_encoder(clin_var)
        # clin_var_encoder = clin_var_encoder.view(-1, 160, 160).unsqueeze(dim=1).unsqueeze(dim=-1).expand_as(img['input'])
        #
        # x = (img['input']*clin_var_encoder).permute(0, 1, 4, 2, 3)
        #
        x = (img['input']).permute(0, 1, 4, 2, 3)
        x = torch.cat([x, task_encoding],1)
        # 将影像数据的通道进行调整，并将任务编码与影像数据合并
        # The channel of image data is adjusted, and task coding is combined with image data

        x_output, hidden_states = self.unetr_pp_encoder(x)
        x_output = einops.rearrange(x_output, "b c h w d -> b (h w d) c")
        # unetr_pp_encoder编码影像数据，获得输出和隐藏状态。随后对输出进行维度调整
        # unetr_pp_encoder encode image data to obtain output and hidden state. The output is then dimensioned

        convBlock = self.encoder1(x)


        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]
        # 获取编码器的四个输出，用于后续的解码操作。
        # Gets the four outputs of the encoder for subsequent decoding operations

        # print('enc1  ', enc1.shape)
        # print('enc2  ', enc2.shape)
        # print('enc3  ', enc3.shape)
        # print('enc4  ', enc4.shape)

        # -------- encoder output
        # enc1_conv = self.conv1(enc1)
        # enc2_conv = self.conv2(enc2)
        # enc4_conv = self.conv4(enc4)
        # enc_conv = self.conv_out(torch.cat([enc1_conv, enc2_conv, enc3, enc4_conv], dim=1))
        # ---------

        enc4 = einops.rearrange(enc4, "b c h w d -> b (h w d) c")
        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        # 将编码器的输出通过一系列解码器进行处理，逐层恢复特征图。
        # The output of the encoder is processed through a series of decoders to recover the feature map layer by layer

        out = self.decoder2(dec1, convBlock)

        out1 = self.out1(out)
        B, C, D, H, W = out1.shape
        up = nn.Upsample(size=(D, H, W))
        # -------- encoder output
        # out1_conv = self.conv_dec(out1).expand_as(enc_conv)

        # out 1 2 3  -> mask
        out1 = out1.permute(0, 1, 3, 4, 2)
        out2 = up(self.out2(dec1)).permute(0, 1, 3, 4, 2)
        out3 = up(self.out3(dec2)).permute(0, 1, 3, 4, 2)

        if self.do_ds:
            logits = torch.stack((out1, out2, out3), dim=0)
        else:
            logits = out1
        # 对解码后的特征图进行处理，生成多尺度的分割结果（logits）。
        # The decoded feature map is processed to generate multi-scale segmentation results (logits).

        # --------------------                          #   MLP
        img_out = out1
        # img_out = out1_conv
        img_out = einops.rearrange(img_out, "b c d h w -> b c (d h w)")
        img_out = torch.mean(img_out, dim=1).squeeze(dim=1)
        img_out = self.img_proj(img_out)  # B 64
        # 将影像输出特征进行投影处理，生成影像特征向量。
        # Image output features are projected to generate image feature vectors.

        # --------------------                          #   RESNET
        # img_out = self.img_resnet(out1)
        # --------------------



        clin_var = self.EHR_proj(clin_var)  # B 64
        # 将临床变量进行投影处理
        # The clinical variables were projected

        mtlr_input = torch.cat([clin_var, img_out], dim=1)  # B 128
        # 将临床变量特征(clin_var)和影像特征(img_out)在特征维度上拼接起来
        # Clinical variable features (clin_var) and image features (img_out) are concatenated in the feature dimension
        mtlr_input = self.mtlr_fc(mtlr_input)
        # 将拼接后的特征向量 mtlr_input 通过多层感知机（MLP）进行处理
        # The splicing feature vector mtlr_input is processed by multi-layer perceptron (MLP)

        # risk_out = self.mtlr(mtlr_input)
        risk_out = self.mtlr(mtlr_input + clin_var)
        # 将 MLP 处理后的特征向量与临床变量特征 clin_var 相加，然后输入到多任务生存风险模型（MTLR）中进行生存风险预测。
        # 如果不使用残差链接模块，则risk_out = self.mtlr(mtlr_input)

        # The feature vector after MLP processing was added with clin_var,
        # and then input into MTLR for survival risk prediction.
        # If you do not use the residual link module, risk_out = self.mtlr(mtlr_input)

        return logits, risk_out         #  logits ->mask    risk_out ->  survival
