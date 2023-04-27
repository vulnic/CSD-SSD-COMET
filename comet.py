import torch
import numpy as np

from torch.nn.parameter import Parameter
import torchvision.transforms.functional as TF
from torch.nn.functional import tanh

from utils.augmentations import augment_bbox

# prior = {
# “rotate”: [0, 0.4], #[mean, std]
# “shear”: [0, 0.05],
# “scale”: [0, 0.25],
# “translate”: [0,.4],
# “h-flip”: [0, 0.01],
# “v-flip”: [-0.8, ]
# }


class COMET(torch.nn.module):
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
            self.prior = {k:torch.FloatTensor(v, device=device) for k,v in prior.items()}
        else:
            # default prior
            prior = {
                "rotate":1.0,
                "shear":1.0,
                "scale":1.0,
                "h-translate":1.0,
                "v-translate":1.0,
                "h-flip":1.0,
                "v-flip":1.0
            }

        self.rotate_weight  = Parameter(torch.FloatTensor([0], device=device)) # Parameter(torch.empty((1,), **factory_kwargs)) 
        self.h_shear_weight = Parameter(torch.FloatTensor([0], device=device)) # Parameter(torch.empty((1,), **factory_kwargs))
        self.v_shear_weight = Parameter(torch.FloatTensor([0], device=device)) # Parameter(torch.empty((1,), **factory_kwargs))
        self.scale_weight   = Parameter(torch.FloatTensor([0], device=device)) # Parameter(torch.empty((1,), **factory_kwargs))
        self.h_translate_weight = Parameter(torch.FloatTensor([0], device=device)) # Parameter(torch.empty((1,), **factory_kwargs))
        self.v_translate_weight = Parameter(torch.FloatTensor([0], device=device)) # Parameter(torch.empty((1,), **factory_kwargs))
        self.h_flip_weight      = Parameter(torch.FloatTensor([0], device=device)) # Parameter(torch.empty((1,), **factory_kwargs))
        self.v_flip_weight      = Parameter(torch.FloatTensor([0], device=device)) # Parameter(torch.empty((1,), **factory_kwargs))
        super.__init__()

    def forward(self,x,boxes):
        h,w = x.size()[-2:] # should be (...,H,W)
        center = [w * 0.5,h * 0.5]

        # first flip the image to not mess with affine transforms
        h_flip_prob = 0.5 * (1.0 + tanh(self.h_flip_weight)) # range = [0,1]
        v_flip_prob = 0.5 * (1.0 + tanh(self.v_flip_weight)) # range = [0,1]
        h_flip_prob = h_flip_prob * self.prior['h-flip'] # range = [0,prior]
        v_flip_prob = v_flip_prob * self.prior['v-flip'] # range = [0,prior]
        if h_flip_prob > 0.5:
            x = TF.hflip(x)
        if v_flip_prob > 0.5:
            x = TF.vflip(x)

        # range = prior*[-1,1] = [-prior,prior]
        absolute_rotate  = self.prior['rotate'] * tanh(self.rotate_weight)
        absolute_h_shear = self.prior['h-shear'] * tanh(self.h_shear_weight)
        absolute_v_shear = self.prior['v-shear'] * tanh(self.v_shear_weight)
        absolute_scale   = self.prior['scale'] * tanh(self.scale_weight)
        h_translate      = self.prior['h-translate'] * tanh(self.h_translate_weight)
        v_translate      = self.prior['v-translate'] * tanh(self.v_translate_weight)

        relative_rotate  = 360 * absolute_rotate
        relative_h_shear = 180 * absolute_h_shear
        relative_v_shear = 180 * absolute_v_shear
        relative_scale   = 0.5 * (absolute_scale + 1)

        inv_relative_rotate  = 360 * -absolute_rotate
        inv_relative_h_shear = 180 * -absolute_h_shear
        inv_relative_v_shear = 180 * -absolute_v_shear
        inv_relative_scale   = 0.5 * (-absolute_scale + 1)
        inv_h_translate   = -h_translate
        inv_v_translate   = -v_translate

        # self.save_params(angle=relative_rotate,
        #                  translate=[absolute_h_translate, absolute_v_translate],
        #                  scale=relative_scale,
        #                  shear=relative_shear)

        # this is the matrix used for transforming the image
        mat = TF._get_inverse_affine_matrix(center,
                                            angle=relative_rotate,
                                            translate=[h_translate, v_translate],
                                            scale=relative_scale,
                                            shear=[relative_h_shear,relative_v_shear],
                                            inverted=False)
        
        inv_mat = TF._get_inverse_affine_matrix(center,
                                                angle=inv_relative_rotate,
                                                translate=[inv_h_translate, inv_v_translate],
                                                scale=inv_relative_scale,
                                                shear=[inv_relative_h_shear,inv_relative_v_shear],
                                                inverted=True)

        # add row to mat
        full_mat = np.array([mat[:3],mat[3:],[0,0,1]],dtype=np.float32)
        inv_full_mat = np.array([inv_mat[:3],inv_mat[3:],[0,0,1]],dtype=np.float32)

        
        assert (boxes[0].shape == (4,)), f"bounding box shape incorrect: boxes[0]={boxes[0]}"
        new_coords = []
        for box in boxes:
            aug_box,coords = augment_bbox(box,full_mat,return_aug_polygon=True)
            new_coords.append(coords)

        x = TF.affine(x,
                      angle=relative_rotate,
                      translate=[h_translate, v_translate],
                      scale=relative_scale,
                      shear=[relative_h_shear,relative_v_shear],
                      center=center)

        return x, np.array(new_coords,dtype=np.float32), inv_full_mat

    # def save_params(self,angle,translate,scale,shear):
    #     self.last_used_params = [angle,translate,scale,shear]


        
    