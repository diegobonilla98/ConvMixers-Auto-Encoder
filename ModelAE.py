import torch
from torch import nn
from torch.autograd import Variable


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(self):
        super(ConvMixer, self).__init__()

        # in_shape = (3, 224, 224)
        h = 768  # 1536
        depth = 15  # 32
        p = 7
        k = 7  # 9

        self.patch_embedding_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=h, kernel_size=p, stride=p),
            nn.GELU(),
            nn.BatchNorm2d(h)
        )

        self.conv_mixer_layer = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(h, h, kernel_size=k, groups=h, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(h)
                )),
                nn.Conv2d(h, h, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(h)
            ) for _ in range(depth)]
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 128, kernel_size=k, stride=p),
            nn.GELU(),
            nn.BatchNorm2d(128),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(128, 1, kernel_size=k, stride=2, padding=3),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        x = self.patch_embedding_net(input_data)
        x = self.conv_mixer_layer(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    model = ConvMixer()
    print(model)
    # model = model.cuda()
    model.eval()
    image = torch.rand((1, 3, 224, 224))  # .cuda()
    image = Variable(image)
    out = model(image)
    print(out)
