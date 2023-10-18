from torch import nn

class Generator(nn.Module):
    """
    Takes in random noise vector of 100 dims and transforms it into a generator image of size 64 X 64 X 3
    # 100x1 → 1024x4x4 → 512x8x8 → 256x16x16 → 128x32x32 → 64x64x3
    """
    def __init__(self, image_shape=(128, 128), zdim=100) -> None:
        super().__init__()
        self.image_shape = image_shape
        self.zdim = zdim
        self.fm_size = 64 # size of the feature maps
        self.out_channels = 3
        self.model = nn.Sequential(

            # Project and Reshape
            # first layer, the noise vector is transformed into a conv block sphaped input using 1023 kernels of size 4X4 with a stride of 1 and 0 padding
            nn.ConvTranspose2d(zdim, self.fm_size * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.fm_size*8), # normalize across the depth
            nn.ReLU(),
            
            # Conv 1
            # We reduce the dimensions of the output to half but increase the size of feature map to twice
            nn.ConvTranspose2d(self.fm_size*8, self.fm_size*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.fm_size*4),
            nn.ReLU(),

            # Conv 2
            # Further reduce the dimensions of the output to half and double the dimensions of the feature maps
            nn.ConvTranspose2d(self.fm_size*4, self.fm_size*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.fm_size*2),
            nn.ReLU(),

            # Conv 3
            nn.ConvTranspose2d(self.fm_size*2, self.fm_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.fm_size),
            nn.ReLU(),

            # Conv 4
            nn.ConvTranspose2d(self.fm_size, self.out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()

        )  

    def forward(self, input_noise):
        return self.model(input_noise)


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        img_channels = 3
        disc_fm_count = 64
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, disc_fm_count, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(disc_fm_count, disc_fm_count*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(disc_fm_count*2),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(disc_fm_count*2, disc_fm_count*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(disc_fm_count*4),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(disc_fm_count*4, disc_fm_count*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(disc_fm_count*8),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(disc_fm_count*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )


    def forward(self, input_img):
        return self.model(input_img)
    

def weights_init(m):
    """
    Initialize weights
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)