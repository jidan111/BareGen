from .structs import *
from ..base_structs import *
from .losses import *


class AutoEncoder(ConfigModule):
    def __init__(self, in_channels, latent_dim, channels, attentions, depth, sample_mode="inter", resnet_num=2,
                 dropout_p=.1, loss_func=None, checkpoint_enable=False):
        super(AutoEncoder, self).__init__()
        if loss_func is None:
            self.loss_func = AutoEncoderKLLoss(using_perception=False, perception_weight=1., perception_net="alex",
                                               kl_weight=1e-6, device=None)
        else:
            self.loss_func = loss_func
        encoder_channels, decoder_channels = get_channels_array(channels, layer_nums=depth)
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=encoder_channels[0][0], kernel_size=1)
        self.encoder = Encoder(channels=encoder_channels, mode=sample_mode, resnet_num=resnet_num,
                               attentions=attentions, dropout_p=dropout_p, checkpoint_enable=checkpoint_enable)
        self.decoder = Decoder(channels=decoder_channels, mode=sample_mode,
                               resnet_num=resnet_num,
                               attentions=attentions[::-1], dropout_p=dropout_p, checkpoint_enable=checkpoint_enable)
        self.q_enc = nn.Conv2d(in_channels=encoder_channels[-1][-1],
                               out_channels=latent_dim * 2, kernel_size=3, stride=1, padding=1)
        self.p_dec = nn.Conv2d(in_channels=latent_dim, out_channels=decoder_channels[0][0],
                               kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(in_channels=decoder_channels[-1][-1], out_channels=in_channels, kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, x):
        x = self.init_conv(x)
        encode = self.encoder(x)
        q = DiagonalGaussianDistribution(self.q_enc(encode))
        latent = q.sample()
        p = self.p_dec(latent)
        decode = self.decoder(p)
        out = self.out_conv(decode)
        return out, q

    def loss(self, x, return_features=False):
        inputs, latent = self(x)
        loss = self.loss_func(inputs, x, latent)
        if return_features:
            return loss, inputs
        return loss

    def get_last_layer_weight(self):
        return self.out_conv.weight

    @torch.no_grad()
    def image2latent(self, x):
        x = self.init_conv(x)
        encode = self.encoder(x)
        q = DiagonalGaussianDistribution(self.q_enc(encode))
        latent = q.mode()
        return latent

    @torch.no_grad()
    def latent2image(self, x):
        p = self.p_dec(x)
        decode = self.decoder(p)
        out = self.out_conv(decode)
        return out


class VQAutoEncoder(ConfigModule):
    def __init__(self, in_channels, embed_num, latent_dim, channels, attentions, depth,
                 sample_mode="inter", resnet_num=2,
                 dropout_p=.1, beta=0.25,
                 use_vq_bridge=True, vq_bridge_mul_size=16, vq_bridge_head_num=4, vq_bridge_layer_num=2,
                 vq_bridge_dropout_p=.1, vq_bridge_type="attn", loss_func=None, checkpoint_enable=False):
        super(VQAutoEncoder, self).__init__()
        if loss_func is None:
            self.loss_func = VQAutoEncoderLoss(using_perception=False, perception_weight=1., perception_net="alex",
                                               book_weight=1., device=None)
        else:
            self.loss_func = loss_func
        encoder_channels, decoder_channels = get_channels_array(channels, layer_nums=depth)
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=encoder_channels[0][0], kernel_size=1)
        self.encoder = Encoder(channels=encoder_channels, mode=sample_mode, resnet_num=resnet_num,
                               attentions=attentions, dropout_p=dropout_p, checkpoint_enable=checkpoint_enable)
        self.decoder = Decoder(channels=decoder_channels, mode=sample_mode,
                               resnet_num=resnet_num,
                               attentions=attentions[::-1], dropout_p=dropout_p, checkpoint_enable=checkpoint_enable)
        self.q_enc = nn.Conv2d(in_channels=encoder_channels[-1][-1],
                               out_channels=latent_dim, kernel_size=3, stride=1, padding=1)
        self.p_dec = nn.Conv2d(in_channels=latent_dim, out_channels=decoder_channels[0][0],
                               kernel_size=3, stride=1, padding=1)
        self.book = VectorQuantizer(embed_num=embed_num, embed_dim=latent_dim, beta=beta,
                                    use_vq_bridge=use_vq_bridge, vq_bridge_mul_size=vq_bridge_mul_size,
                                    vq_bridge_head_num=vq_bridge_head_num,
                                    vq_bridge_layer_num=vq_bridge_layer_num,
                                    vq_bridge_dropout_p=vq_bridge_dropout_p, vq_bridge_type=vq_bridge_type)
        self.out_conv = nn.Conv2d(in_channels=decoder_channels[-1][-1], out_channels=in_channels, kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, x):
        x = self.init_conv(x)
        encode = self.encoder(x)
        q = self.q_enc(encode)
        z, book_loss, index = self.book(q)
        p = self.p_dec(z)
        decode = self.decoder(p)
        out = self.out_conv(decode)
        return out, book_loss

    def loss(self, x, return_features=False):
        inputs, book_loss = self(x)
        loss = self.loss_func(inputs, x, book_loss)
        if return_features:
            return loss, inputs
        return loss

    def get_last_layer_weight(self):
        return self.out_conv.weight

    @torch.no_grad()
    def image2latent(self, x):
        x = self.init_conv(x)
        encode = self.encoder(x)
        q = self.q_enc(encode)
        latent = self.book.feature2latent(q)
        return latent

    @torch.no_grad()
    def latent2image(self, x):
        p = self.p_dec(x)
        decode = self.decoder(p)
        out = self.out_conv(decode)
        return out


class PlainAutoEncoder(ConfigModule):
    def __init__(self, in_channels, latent_dim, channels, attentions, depth, sample_mode="inter", resnet_num=2,
                 dropout_p=.1, loss_func=None, checkpoint_enable=False):
        super(PlainAutoEncoder, self).__init__()
        if loss_func is None:
            self.loss_func = PlainAutoEncoderLoss(using_perception=False, perception_weight=1., perception_net="alex",
                                                  device=None)
        else:
            self.loss_func = loss_func
        encoder_channels, decoder_channels = get_channels_array(channels, layer_nums=depth)
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=encoder_channels[0][0], kernel_size=1)
        self.encoder = Encoder(channels=encoder_channels, mode=sample_mode, resnet_num=resnet_num,
                               attentions=attentions, dropout_p=dropout_p, checkpoint_enable=checkpoint_enable)
        self.decoder = Decoder(channels=decoder_channels, mode=sample_mode,
                               resnet_num=resnet_num,
                               attentions=attentions[::-1], dropout_p=dropout_p, checkpoint_enable=checkpoint_enable)
        self.q_enc = nn.Conv2d(in_channels=encoder_channels[-1][-1],
                               out_channels=latent_dim, kernel_size=3, stride=1, padding=1)
        self.p_dec = nn.Conv2d(in_channels=latent_dim, out_channels=decoder_channels[0][0],
                               kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(in_channels=decoder_channels[-1][-1], out_channels=in_channels, kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, x):
        x = self.init_conv(x)
        encode = self.encoder(x)
        latent = self.q_enc(encode)
        p = self.p_dec(latent)
        decode = self.decoder(p)
        out = self.out_conv(decode)
        return out

    def loss(self, x, return_features=False):
        inputs = self(x)
        loss = self.loss_func(inputs, x)
        if return_features:
            return loss, inputs
        return loss

    def get_last_layer_weight(self):
        return self.out_conv.weight

    @torch.no_grad()
    def image2latent(self, x):
        x = self.init_conv(x)
        encode = self.encoder(x)
        latent = self.q_enc(encode)
        return latent

    @torch.no_grad()
    def latent2image(self, x):
        p = self.p_dec(x)
        decode = self.decoder(p)
        out = self.out_conv(decode)
        return out
