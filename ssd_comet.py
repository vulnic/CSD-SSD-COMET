from ssd import SSD, build_ssd
import torch
from utils.augmentations import augment_bboxes
# prior = {
# “rotate”: [0, 0.4], #[mean, std]
# “shear”: [0, 0.05],
# “scale”: [0, 0.25],
# “translate”: [0,.4],
# “h-flip”: [0, 0.01]
# }


class SSD_COMET(torch.nn.Module):
    def __init__(self,comet_net,ssd_net):
        super().__init__()
        self.comet_net = comet_net
        self.ssd_net = ssd_net
    def forward(self,images,boxes):
      aug_images, inv_mat = self.comet_net(images, boxes)
      out, loc, conf, features = self.ssd_net(images)
      _, loc_aug, conf_aug, features_aug = self.ssd_net(aug_images)

      inv_locs = augment_bboxes(loc_aug,inv_mat,cuda=True)

      return out, inv_locs, loc, conf, features, loc_aug, conf_aug, features_aug