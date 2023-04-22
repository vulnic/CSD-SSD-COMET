# prior = {
# “rotate”: [0, 0.4], #[mean, std]
# “shear”: [0, 0.05],
# “scale”: [0, 0.25],
# “translate”: [0,.4],
# “h-flip”: [0, 0.01],
# “v-flip”: [-0.8, ]
# }

import torch
from torch.nn.parameter import Parameter
from torchvision.transforms.functional as TF
from torch.nn.functional import tanh

class COMET(nn.module):
    def __init__(self, device=None, dytpe=None, prior=None):
        # factory_kwargs = {'device': device, 'dtype': dtype}
        if prior:
            assert "rotate"      in prior.keys()
            assert "shear"       in prior.keys()
            assert "scale"       in prior.keys()
            assert "h-translate" in prior.keys()
            assert "v-translate" in prior.keys()
            assert "h-flip"      in prior.keys()
            assert "v-flip"      in prior.keys()
            prior = {k:FloatTensor(v, device=device) for k,v in prior.items()}

        self.rotate_weight    = Parameter(torch.FloatTensor([0], device=device)) # Parameter(torch.empty((1,), **factory_kwargs)) 
        self.shear_weight     = Parameter(torch.FloatTensor([0], device=device)) # Parameter(torch.empty((1,), **factory_kwargs))
        self.scale_weight     = Parameter(torch.FloatTensor([0], device=device)) # Parameter(torch.empty((1,), **factory_kwargs))
        self.h_translate_weight = Parameter(torch.FloatTensor([0], device=device)) # Parameter(torch.empty((1,), **factory_kwargs))
        self.v_translate_weight = Parameter(torch.FloatTensor([0], device=device)) # Parameter(torch.empty((1,), **factory_kwargs))
        self.h_flip_weight      = Parameter(torch.FloatTensor([0], device=device)) # Parameter(torch.empty((1,), **factory_kwargs))
        self.v_flip_weight      = Parameter(torch.FloatTensor([0], device=device)) # Parameter(torch.empty((1,), **factory_kwargs))
        super.__init__()

    def forward(self,x):
        h,w = x.size()[-2:] # should be (...,H,W)
        center = [w * 0.5,h * 0.5]

        # first flip the image to not mess with affine transforms
        h_flip_prob = 0.5 * (1.0 + tanh(self.h_flip_weight)) # range = [0,1]
        v_flip_prob = 0.5 * (1.0 + tanh(self.v_flip_weight)) # range = [0,1]
        h_flip_prob = h_flip_prob * prior['h-flip'] # range = [0,prior]
        v_flip_prob = v_flip_prob * prior['v-flip'] # range = [0,prior]
        if h_flip_prob > 0.5:
            x = TF.hflip(x)
        if v_flip_prob > 0.5:
            x = TF.vflip(x)

        # range = prior*[-1,1] = [-prior,prior]
        absolute_rotate      = prior['rotate'] * tanh(self.rotate_weight)
        absolute_shear       = prior['shear'] * tanh(self.shear_weight)
        absolute_scale       = prior['scale'] * tanh(self.scale_weight)
        absolute_h_translate = prior['h-translate'] * tanh(self.h_translate_weight)
        absolute_v_translate = prior['v-translate'] * tanh(self.v_translate_weight)

        relative_rotate = 360 * absolute_rotate
        relative_shear = 180 * absolute_shear
        relative_scale = 0.5 * (absolute_scale + 1)

        self.save_params(angle=relative_rotate,
                         translate=[absolute_h_translate, absolute_v_translate],
                         scale=relative_scale,
                         shear=relative_shear)

        x = TF.affine(x,
                      angle=relative_rotate,
                      translate=[absolute_h_translate, absolute_v_translate]
                      scale=relative_scale
                      shear=relative_shear)

        return x

    def save_params(self,angle,translate,scale,shear):
        self.last_used_params = [angle,translate,scale,shear]


        
    