from .openaimodel import TimestepBlock, TimestepEmbedSequential, ResBlock, Upsample, Downsample, SpatialTransformer
from torch import nn
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.util import instantiate_from_config
from copy import deepcopy
import torch as th
import random


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            num_heads=8,
            use_scale_shift_norm=False,
            transformer_depth=1,
            context_dim=None,
            fuser_type=None,
            inpaint_mode=False,
            grounding_downsampler=None,

    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.context_dim = context_dim
        self.fuser_type = fuser_type
        self.inpaint_mode = inpaint_mode
        self.dims=dims
        assert fuser_type in ["gatedSA", "gatedSA2", "gatedCA"]


        time_embed_dim = model_channels * 4   #时间嵌入的维度为模型嵌入的4倍
        self.time_embed = nn.Sequential(                    #创建时间嵌入
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.downsample_net = None   #这个下采样网络并不是UNet模型中的下采样，而是对于需要再次卷积操作的额外输入的下采样（如canny边缘图片）
        self.additional_channel_from_downsampler = 0  #额外下采样产生的多出来的通道数
        self.first_conv_type = "SD"   #第一个卷积的类别 应该是用于inpaint的卷积
        self.first_conv_restorable = True
        if grounding_downsampler is not None:   #输入的控制信息是否需要再次下采样
            self.downsample_net = instantiate_from_config(grounding_downsampler)
            self.additional_channel_from_downsampler = self.downsample_net.out_dim
            self.first_conv_type = "GLIGEN"

        if inpaint_mode:  #如果是inpaint模式的话（局部修改）
            # The new added channels are: masked image (encoded image) and mask, which is 4+1
            in_c = in_channels + self.additional_channel_from_downsampler + in_channels + 1  #如果是inpaint模式输入的通道数要增加，因为有新的信息加入进来并在通道维度上进行拼接，之后要使用一个额外的卷积进行通道数的恢复
            self.first_conv_restorable = False  # in inpaint; You must use extra channels to take in masked real image
        else:
            in_c = in_channels + self.additional_channel_from_downsampler   #如果不是inpaint模式，只需要在通道数上加上额外的信息的通道数

        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_c, model_channels, 3, padding=1))])  #其实输入变成多少通道数的都没问题，因为终究都会通过一个卷积转化为原来的模型通道数
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])  # 第一个zero convolution用于处理control图像的特征
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        self.ground_zero_convs = nn.ModuleList([])
        # = = = = = = = = = = = = = = = = = = = = Down Branch = = = = = = = = = = = = = = = = = = = = #
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch,  #ResBlock只接受CHW的隐变量z和时间嵌入emd，不接受context和objs，所以objs想要进入块中计算只能在transformer内部
                                   time_embed_dim,
                                   dropout,
                                   out_channels=mult * model_channels,
                                   dims=dims,
                                   use_checkpoint=use_checkpoint,
                                   use_scale_shift_norm=use_scale_shift_norm, )]

                ch = mult * model_channels
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(SpatialTransformer(ch, key_dim=context_dim, value_dim=context_dim, n_heads=num_heads,
                                                     d_head=dim_head, depth=transformer_depth, fuser_type=fuser_type,
                                                     use_checkpoint=use_checkpoint)) #做交叉注意力机制时，想更新谁的值谁就是query，参考的作为key

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                # print(f"in {level} level {_} blocks,create a zero block {ch}")
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:  # will not go to this downsample branch in the last feature
                out_ch = ch                     # 注意，下采样只会对z的大小尺寸进行改变，不会对其他embedding的维度进行改变
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                # print(f"in {level} level downsample ,create a zero block {ch}")
                ds *= 2

        dim_head = ch // num_heads

        # self.input_blocks = [ C |  RT  RT  D  |  RT  RT  D  |  RT  RT  D  |   R  R   ]

        # = = = = = = = = = = = = = = = = = = = = BottleNeck = = = = = = = = = = = = = = = = = = = = #

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch,
                     time_embed_dim,
                     dropout,
                     dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm),
            SpatialTransformer(ch, key_dim=context_dim, value_dim=context_dim, n_heads=num_heads, d_head=dim_head,
                               depth=transformer_depth, fuser_type=fuser_type, use_checkpoint=use_checkpoint),
            ResBlock(ch,
                     time_embed_dim,
                     dropout,
                     dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm))
        self.middle_block_out=self.make_zero_conv(ch)
        # 这里没有输出的blocks

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def restore_first_conv_from_SD(self):
        if self.first_conv_restorable:
            device = self.input_blocks[0][0].weight.device

            SD_weights = th.load("SD_input_conv_weight_bias.pth")
            self.GLIGEN_first_conv_state_dict = deepcopy(self.input_blocks[0][0].state_dict())

            self.input_blocks[0][0] = conv_nd(2, 4, 320, 3, padding=1)
            self.input_blocks[0][0].load_state_dict(SD_weights)
            self.input_blocks[0][0].to(device)

            self.first_conv_type = "SD"
        else:
            print(
                "First conv layer is not restorable and skipped this process, probably because this is an inpainting model?")

    def restore_first_conv_from_GLIGEN(self):
        breakpoint()  # TODO

    def forward(self, input, objs):

        # Time embedding
        t_emb = timestep_embedding(input["timesteps"], self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        # input tensor
        h = input["x"]
        if self.downsample_net != None and self.first_conv_type == "GLIGEN": #这里是用于处理有额外输入的控制信息，如果只是坐标和图片则不会用到这里
            temp = self.downsample_net(input["grounding_extra_input"])
            h = th.cat([h, temp], dim=1)
        if self.inpaint_mode:
            if self.downsample_net != None:
                breakpoint()  # TODO: think about this case
            h = th.cat([h, input["inpainting_extra_input"]], dim=1)

        # Text input
        context = input["context"]

        # Start forwarding
        hs = []

        for module,zero_conv in zip(self.input_blocks,self.zero_convs):
            h = module(h, emb, context, objs)
            hs.append(zero_conv(h,emb,context,objs))

        h = self.middle_block(h, emb, context, objs)

        hs.append(self.middle_block_out(h,emb,context,objs))


        return hs