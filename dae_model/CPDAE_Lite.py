#!/bin/python3
from ptflops import get_model_complexity_info
from torch import nn
from dae_model.lib.pixelshuffle1d import PixelShuffle1D, PixelUnshuffle1D
from torchinfo import summary


class res_1d(nn.Module):
    def __init__(self, io_ch, ks=3):
        super(res_1d, self).__init__()
        assert (ks - 1) % 2 == 0
        pd = int((ks - 1) / 2)
        self.res1_3 = nn.Sequential(
            nn.Conv1d(io_ch, io_ch, ks, padding=pd),
            nn.ReLU(True),
            nn.Conv1d(io_ch, io_ch, ks, padding=pd))
        self.res_relu5 = nn.ReLU(True)

    def forward(self, x):
        return self.res_relu5(x + self.res1_3(x))

class CPDAE_Lite(nn.Module):
    def __init__(self):
        super(CPDAE_Lite, self).__init__()

        self.encode_stage = nn.Sequential()
        self.decode_stage = nn.Sequential()
        self.short_pwconv = nn.Sequential()

        """input-layer"""
        self.encode_stage.add_module("input", nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=1),
            res_1d(16, 5)
        ))

        """encoder"""
        self.encode_stage.add_module("encoder1",
                                     nn.Sequential(
                                         res_1d(16, 5),
                                         PixelUnshuffle1D(2),
                                         nn.Conv1d(32, 16, 1)
                                     )
                                     )

        self.encode_stage.add_module("encoder2",
                                     nn.Sequential(
                                         res_1d(16, 5),
                                         PixelUnshuffle1D(2),
                                         nn.Conv1d(32, 16, 1)
                                     )
                                     )

        self.encode_stage.add_module("encoder3",
                                     nn.Sequential(
                                         res_1d(16, 5),
                                         PixelUnshuffle1D(2),
                                         nn.Conv1d(32, 16, 1)
                                     )
                                     )

        self.encode_stage.add_module("encoder4",
                                     nn.Sequential(
                                         res_1d(16, 5),
                                         PixelUnshuffle1D(2),
                                         nn.Conv1d(32, 16, 1)
                                     )
                                     )
        self.encode_stage.add_module("encoder5",
                                     nn.Sequential(
                                         res_1d(16, 5),
                                         PixelUnshuffle1D(2),
                                         nn.Conv1d(32, 16, 1)
                                     )
                                     )
        self.encode_stage.add_module("encoder6",
                                     nn.Sequential(
                                         res_1d(16, 5),
                                         PixelUnshuffle1D(2),
                                         nn.Conv1d(32, 16, 1)
                                     )
                                     )
        self.encode_stage.add_module("encoder7",
                                     nn.Sequential(
                                         res_1d(16, 5),
                                         PixelUnshuffle1D(2),
                                         nn.Conv1d(32, 16, 1)
                                     )
                                     )
        self.encode_stage.add_module("encoder8",
                                     nn.Sequential(
                                         res_1d(16, 5),
                                         PixelUnshuffle1D(2),
                                         nn.Conv1d(32, 16, 1)
                                     )
                                     )

        """decoder"""
        self.decode_stage.add_module("decoder8",
                                     nn.Sequential(
                                         res_1d(16, 5),
                                         nn.Conv1d(16, 32, 1),
                                         PixelShuffle1D(2),
                                     ))
        self.short_pwconv.add_module("short_pwconv7", nn.Conv1d(1, 16, 1))
        self.decode_stage.add_module("decoder7",
                                     nn.Sequential(
                                         res_1d(16, 5),
                                         nn.Conv1d(16, 32, 1),
                                         PixelShuffle1D(2),
                                     ))

        self.short_pwconv.add_module("short_pwconv6", nn.Conv1d(1, 16, 1))
        self.decode_stage.add_module("decoder6",
                                     nn.Sequential(
                                         res_1d(16, 5),
                                         nn.Conv1d(16, 32, 1),
                                         PixelShuffle1D(2),
                                     ))
        self.short_pwconv.add_module("short_pwconv5", nn.Conv1d(1, 16, 1))
        self.decode_stage.add_module("decoder5",
                                     nn.Sequential(
                                         res_1d(16, 5),
                                         nn.Conv1d(16, 32, 1),
                                         PixelShuffle1D(2),
                                     ))
        self.short_pwconv.add_module("short_pwconv4", nn.Conv1d(1, 16, 1))
        self.decode_stage.add_module("decoder4",
                                     nn.Sequential(
                                         res_1d(16, 5),
                                         nn.Conv1d(16, 32, 1),
                                         PixelShuffle1D(2),
                                     ))
        self.short_pwconv.add_module("short_pwconv3", nn.Conv1d(1, 16, 1))
        self.decode_stage.add_module("decoder3",
                                     nn.Sequential(
                                         res_1d(16, 5),
                                         nn.Conv1d(16, 32, 1),
                                         PixelShuffle1D(2),
                                     ))
        self.short_pwconv.add_module("short_pwconv2", nn.Conv1d(1, 16, 1))
        self.decode_stage.add_module("decoder2",
                                     nn.Sequential(
                                         res_1d(16, 5),
                                         nn.Conv1d(16, 32, 1),
                                         PixelShuffle1D(2),
                                     ))
        self.short_pwconv.add_module("short_pwconv1", nn.Conv1d(1, 16, 1))
        self.decode_stage.add_module("decoder1",
                                     nn.Sequential(
                                         res_1d(16, 5),
                                         nn.Conv1d(16, 32, 1),
                                         PixelShuffle1D(2),
                                     ))
        """output-layer"""
        self.decode_stage.add_module("output",
                                     nn.Sequential(
                                         res_1d(16, 5),
                                         nn.Conv1d(16, 1, 1),
                                     )
                                     )

        self.CWAP = nn.Conv1d(16, 1, 1, bias=False)
        self.CWAP.weight.data.fill_(1 / 16)
        self.CWAP.requires_grad_(False)

    def forward(self, x):

        avg_arr = []
        """input"""
        x = self.encode_stage[0](x)
        """"encoder"""
        for i in range(1, len(self.encode_stage)):
            x = self.encode_stage[i](x)
            avg_arr.append(self.CWAP(x))
            # avg_arr.append(torch.mean(x, dim=1).view(x.shape[0], 1, -1)) # Channel-wise average pooling
        """decoder"""
        x = self.decode_stage[0](x) # decoder8, no need to add the tensor from cwap
        for i in range(1, len(self.decode_stage)-1):
            short = self.short_pwconv[i-1](avg_arr[len(avg_arr)-i-1])
            x = self.decode_stage[i](x + short)
        """output"""
        x = self.decode_stage[-1](x)
        return x

    def get_model_net(self):
        return "CPDAE_Lite"


if __name__ == "__main__":
    model = CPDAE_Lite().cuda()
    print(summary(model, (1, 1, 1024), device="cuda", verbose=0).__repr__())
    macs, params = get_model_complexity_info(model, (1, 1024), as_strings=False,
                                             print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
