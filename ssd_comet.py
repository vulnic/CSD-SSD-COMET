from ssd import SSD, build_ssd
# prior = {
# “rotate”: [0, 0.4], #[mean, std]
# “shear”: [0, 0.05],
# “scale”: [0, 0.25],
# “translate”: [0,.4],
# “h-flip”: [0, 0.01]
# }


class COMET_SSD(nn.module):
    def __init__(self,prior):
        comet = COMET(prior)
        ssd = build_ssd('train', cfg['min_dim'], cfg['num_classes'], return_loc_conf=True)
        super().__init__()
    def forward(self,x):
        x_aug = comet(x)
        aug_params = comet.last_used_params() # [angle,translate,scale,shear] := [float,List[float],float,float]
        out, loc, conf = ssd(x)
          _, loc_aug, conf_aug = ssd(x_aug)
        return out, conf, conf_aug, loc, loc_aug, aug_params