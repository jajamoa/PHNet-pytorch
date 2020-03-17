import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
FRAMES_PER_IMG = 3


class PHNet(nn.Module):
    def __init__(self,load_weights=False):
        super(PHNet, self).__init__()
        self.seen = 0
        self.frame = 3
        self.conv3d = nn.Conv3d(3, 3, kernel_size=3, padding=1,  stride=(1,1,1))
        self.conv11 = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(self.frame + 1,1,1), stride=1)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend_feat2 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]
        self.backend_feat2  = [512, 512, 256, 128]
        self.frontend = make_layers(self.frontend_feat)
        self.frontend2 = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels = 512, dilation = 2)
        self.backend2 = make_layers(self.backend_feat2, in_channels = 512, dilation = 2)
        self.output_layer = nn.Conv2d(192, 1, kernel_size=1)
        self.CRPool1 = nn.AvgPool3d(kernel_size=(2, 3, 3), stride=1, padding=(1, 1, 1))
        self.CRPool2 = nn.AvgPool3d(kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        self.BN = nn.BatchNorm3d(3, affine=True, momentum=.95)
        self.BN2 = nn.BatchNorm2d(64, affine=True, momentum=.95)
        self.relu = nn.ReLU()
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            #num = 0
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
                list(self.frontend2.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

    def forward(self,x):
        y = x.clone()
        y = self.CRPool1(y)
        y = self.conv11(y*y)
        y = torch.squeeze(y, dim = 2)
        y = self.frontend2(y)
        y = self.backend2(y)
        x = x[:,:,-1,:,:] 
        x = self.frontend(x)
        x = self.backend(x)
        x = torch.cat([x,y], dim=1)
        x = self.output_layer(x)
        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=0.01)


def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
                #layers.append(CBAM(v, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False))
            in_channels = v
    return nn.Sequential(*layers)

def main():
    model=PHNet()
    x = torch.rand(2, 5, 600, 400)
    x = model(x)
    print(x.shape)

if __name__ == '__main__':
    main()
