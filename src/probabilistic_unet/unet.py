from .unet_blocks import *


class Unet(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels, num_classes, num_filters, initializers, apply_last_layer=True, padding=True, norm=False, mc_dropout=False, dropout_rate=0.0, save_decoder_features=False):
        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.contracting_path = nn.ModuleList()
        self.save_decoder_features = save_decoder_features

        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.contracting_path.append(DownConvBlock(input,
                                                       output,
                                                       initializers,
                                                       padding,
                                                       pool=pool,
                                                       norm=norm,
                                                       mc_dropout=mc_dropout,
                                                       dropout_rate=dropout_rate))


        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 2

        for i in range(n, -1, -1):
            input = output + self.num_filters[i]
            output = self.num_filters[i]

            if i == 0:
                norm = False
            else:
                norm = norm

            self.upsampling_path.append(UpConvBlock(input,
                                                    output,
                                                    initializers,
                                                    padding,
                                                    norm=norm,
                                                    mc_dropout=mc_dropout,
                                                    dropout_rate=dropout_rate))

        if self.apply_last_layer:
            self.last_layer = nn.Conv3d(output, num_classes, kernel_size=1)

        if self.save_decoder_features:
            self.decoder_heads = nn.ModuleList([
                nn.Conv3d(num_filters, self.num_classes, kernel_size=1)
                for num_filters in self.num_filters[::-1]
            ])

    def forward(self, x):
        encoder_features = []
        decoder_features = []

        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path)-1:
                encoder_features.append(x)

        decoder_features.append(x)
        for i, up in enumerate(self.upsampling_path):
            x = up(x, encoder_features[-i-1])
            decoder_features.append(x)

        if self.training and self.save_decoder_features:
            self.decoder_features = decoder_features
        elif hasattr(self, "decoder_features"):
            del self.decoder_features

        del encoder_features

        if self.apply_last_layer:
            x = self.last_layer(x)

        return x

    def process_decoder_features(self, patch_size):
        upsample = torch.nn.Upsample(patch_size)
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.decoder_features = [
            upsample(logsoftmax(head(feat)))
            for feat, head in zip(self.decoder_features, self.decoder_heads)
        ]
        self.decoder_features = self.decoder_features[::-1]
