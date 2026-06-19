import torch
import torch.nn as nn
import torchvision
from ..util.layers import Reshape

# Encoder / Decoder ################################################################
class EncoderDecoderMixin:
    def fdim(self):
        # Returns the recorded latent (feature) dimension.
        return int(self.parameters["feature_dim"])

    def _build_around(self, state_input_shape):
        # Build encoder and decoder networks.
        self.feature_encoder = self.build_encoder(state_input_shape)
        self.feature_decoder = self.build_decoder(state_input_shape)

        # Call to a method from the superclass (if applicable).
        super()._build_around(state_input_shape)

    def build_encoder(self, input_shape):
        # Unpack the list of encoder modules into nn.Sequential.
        return nn.Sequential(*self._build_encoder(input_shape))

    def build_decoder(self, input_shape):
        # Unpack the list of decoder modules into nn.Sequential.
        return nn.Sequential(*self._build_decoder(input_shape))


class ResNetMixin(EncoderDecoderMixin):
    def autocrop_dimensions(self, input_shape, base=32):
        """
        Given an input shape (C, H, W), compute the extra height and width
        needed so that H and W become multiples of 'base'.

        Args:
            input_shape (tuple): (C, H, W)
            base (int): The number to which height and width should be a multiple.

        Returns:
            dH, dW (int, int): The additional height and width needed.
                               (These are to be added at the bottom/right.)
        """
        C, H, W = input_shape
        dH = (base - (H % base)) % base  # if H is a multiple, dH will be 0
        dW = (base - (W % base)) % base  # if W is a multiple, dW will be 0
        return dH, dW

    def _build_encoder(self, input_shape):
        """
        Build a ResNet-18 encoder that outputs a latent vector.

        We modify the ResNet-18 backbone by replacing its fully connected (fc) layer with an identity.
        This way, the backbone's default forward method (which still applies average pooling and flattening)
        produces a latent vector of dimension self.parameters["feature_dim"] (typically 512).

        However, to build the decoder properly, we need to know the spatial dimensions of the feature map
        produced by the convolutional layers before the average pooling is applied. To obtain this, we run
        a dummy forward pass manually through the backbone up to layer4. The recorded shape (channels, H_enc, W_enc)
        is then stored in self.encoder_feature_shape and its channel count is used as the latent dimension.
        """
        self.input_channels = input_shape[0]  # e.g., (C, H, W)
        print("Building a convolutional encoder using ResNet-18")

        # Load ResNet-18 (set pretrained=True if desired, but note the conv1 channel modification)
        feature_encoder = torchvision.models.resnet18(weights=None)  # or use weights=... if you want pretrained

        # If the number of input channels is not 3, modify the first conv layer accordingly.
        if self.input_channels != 3:
            feature_encoder.conv1 = nn.Conv2d(self.input_channels, 64,
                                       kernel_size=7, stride=2, padding=3, bias=False)

        # The backbone's forward method is: conv1 -> bn1 -> relu -> maxpool -> layers -> avgpool -> flatten -> fc.
        # Run a dummy forward pass through the backbone up to layer4 to record the spatial dimensions.
        # This pass stops before avgpool so that we capture the full spatial feature map.
        dummy_input = torch.zeros(1, *input_shape)
        with torch.no_grad():
            x = feature_encoder.conv1(dummy_input)
            x = feature_encoder.bn1(x)
            x = feature_encoder.relu(x)
            x = feature_encoder.maxpool(x)
            x = feature_encoder.layer1(x)
            x = feature_encoder.layer2(x)
            x = feature_encoder.layer3(x)
            dummy_features = feature_encoder.layer4(x)
        # For a standard 224x224 input, dummy_features.shape might be (1, 512, 7, 7).
        # This shape will adjust automatically for other input sizes.
        self.encoder_feature_shape = dummy_features.shape[1:]  # e.g., (channels_enc, H_enc, W_enc)
        print("Encoder feature shape (from layer4, before avgpool):", self.encoder_feature_shape)

        # Replace the fc layer with a linear layer that outputs the feature dimension.
        feature_encoder.fc = nn.Linear(self.encoder_feature_shape[0], self.fdim())

        # Return the encoder module: the modified ResNet-18 backbone.
        # Its forward method applies avgpool and flattening internally, yielding a latent vector of
        # dimension self.parameters["feature_dim"].
        dH, dW = self.autocrop_dimensions(input_shape)
        return [nn.ZeroPad2d((0, dW, 0, dH)), feature_encoder]

    def _build_decoder(self, input_shape):
        """
        Build a decoder that takes a latent vector (of dimension self.parameters["feature_dim"])
        and reconstructs an image.

        Instead of hardcoding the latent dimension (e.g. 512), we use the stored encoder
        feature shape to determine the mapping. The decoder first maps the latent vector into
        a flattened version of the encoder's feature map, reshapes it, and then uses a series of
        transpose convolutions and an explicit upsampling layer to match the original image size.
        """
        C, H, W = input_shape

        # Retrieve the recorded encoder feature shape.
        channels_enc, H_enc, W_enc = self.encoder_feature_shape

        feature_decoder = nn.Sequential(
            # Map the latent vector to the encoder feature shape.
            nn.Linear(self.fdim(), self.encoder_feature_shape[0]),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_feature_shape[0], channels_enc * H_enc * W_enc),
            nn.ReLU(inplace=True),
            Reshape(channels_enc, H_enc, W_enc),

            # Series of ConvTranspose2d layers to progressively upsample.
            nn.ConvTranspose2d(channels_enc, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Map to the original number of channels.
            nn.ConvTranspose2d(32, C, kernel_size=4, stride=2, padding=1),
        )

        dH, dW = self.autocrop_dimensions(input_shape)
        return [feature_decoder, nn.ZeroPad2d((0, -dW, 0, -dH))]