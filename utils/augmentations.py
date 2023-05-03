import torch
from torchvision import transforms
from torchvision.transforms import Normalize, ToTensor, ColorJitter, Pad, RandomResizedCrop, RandomHorizontalFlip
import cv2
import numpy as np
import types
from numpy import random

def np_augment_bbox(bbox,transform_mat,return_aug_polygon=False):
    # augment_bbox() can take in bbox OR coords
    #  - for augmenting a bbox, use bbox
    #  - for inverting an augmented bbox, use coords
    
    # augment bbox
    # - - - - - - 
    
    if bbox.shape == (4,):
        # augment the bbox
        x1,y1,x2,y2 = bbox
        upper_left  = np.asarray([x1,y1,1],dtype=np.float32)
        upper_right = np.asarray([x2,y1,1],dtype=np.float32)
        lower_left  = np.asarray([x1,y2,1],dtype=np.float32)
        lower_right = np.asarray([x2,y2,1],dtype=np.float32)
    elif bbox.shape == (5,2) or bbox.shape == (4,2):
        # reverse the bbox's augmentation
        # bbox is [ul, ur, lr, ll, ul] or missing last "ul"
        upper_left  = np.asarray(np.append(bbox[0],1),dtype=np.float32)
        upper_right = np.asarray(np.append(bbox[1],1),dtype=np.float32)
        lower_right = np.asarray(np.append(bbox[2],1),dtype=np.float32)
        lower_left  = np.asarray(np.append(bbox[3],1),dtype=np.float32)
    else:
        raise Exception("bbox shape is not compatible", bbox.shape)
    
    # print(upper_left, upper_left.shape, bbox[0])
    new_ul = np.matmul(transform_mat,upper_left)[:2]
    new_ur = np.matmul(transform_mat,upper_right)[:2]
    new_ll = np.matmul(transform_mat,lower_left)[:2]
    new_lr = np.matmul(transform_mat,lower_right)[:2]
    
    coords = [new_ul, new_ur, new_lr, new_ll]
    five_coords = coords + [new_ul] # append for closer loop
    # xs,ys = zip(*five_coords)
    # new_coords = 

    # create surrounding bbox
    # - - - - - - - - - - - -
    npcoords = np.asarray(five_coords,dtype=np.float32)
    all_x_coords = npcoords[:,0]
    all_y_coords = npcoords[:,1]
    
    full_x1 = np.min(all_x_coords)
    full_x2 = np.max(all_x_coords)
    full_y1 = np.min(all_y_coords)
    full_y2 = np.max(all_y_coords)
    
    aug_bbox = np.asarray([full_x1, full_y1, full_x2, full_y2], dtype=np.float32)
    
    if return_aug_polygon:
        return aug_bbox,npcoords
    
    return aug_bbox

def augment_bboxes(bboxes,mat,cuda=False):
    
    # bboxes.shape == (num_batches,num_boxes,4)   = (nB,nBb,4)
    # bboxes.shape == (num_batches,num_boxes,4,2) = (nB,nBb,4,2)
    # bboxes.shape == (num_batches,num_boxes,5,2) = (nB,nBb,5,2)

    if bboxes.shape[2:] == (4,):
        # augment bbox with "mat"

        xA = torch.FloatTensor([[1,0,0,0],
                                [0,1,0,0],
                                [0,1,0,0],
                                [1,0,0,0]])
        
        yA = torch.FloatTensor([[0,0,1,0],
                                [0,0,1,0],
                                [0,0,0,1],
                                [0,0,0,1]])

        ones = torch.ones(bboxes.shape + (1,)) # (nB,nBb,4,1)

        if 'cuda' in str(bboxes.device):
            xA   = xA.clone().cuda()
            yA   = yA.clone().cuda()
            ones = ones.clone().cuda()
            mat  = mat.clone().cuda()

        x_vals = torch.matmul(bboxes,xA)          # (nB,nBb,4)
        x_vals = x_vals.view(bboxes.shape + (1,)) # (nB,nBb,4,1)
        y_vals = torch.matmul(bboxes,yA)          # (nB,nBb,4)
        y_vals = y_vals.view(bboxes.shape + (1,)) # (nB,nBb,4,1)

        coords     = torch.cat([x_vals,y_vals,ones],axis=3) # (nB,nBb,4,3)
        aug_coords = torch.matmul(coords, mat)              # (nB,nBb,4,3)
        aug_coords = coords[:,:,:,:-1] # remove ones column # (nB,nBb,4,2)
        
        # new_x_vals = coords[:,:,:,0].view(bboxes.shape + (1,)) # (nB,nBb,4,1)
        # new_y_vals = coords[:,:,:,1].view(bboxes.shape + (1,)) # (nB,nBb,4,1)
        x_max = torch.max(aug_coords[:,:,:,0],axis=2)[0] # (nB,nBb)
        x_min = torch.min(aug_coords[:,:,:,0],axis=2)[0] # (nB,nBb)
        y_max = torch.max(aug_coords[:,:,:,1],axis=2)[0] # (nB,nBb)
        y_min = torch.min(aug_coords[:,:,:,1],axis=2)[0] # (nB,nBb)

        # this should be x1,y1,x2,y2
        aug_bboxes = torch.stack([x_min,y_min,x_max,y_max],axis=2) # (nB,nBb,4)

        # ul = x1,y1
        # ur = x2,y1
        # lr = x2,y2
        # ll = x1,y2
    
    # elif bboxes.shape[2:] == (4,2) or \
    #      bboxes.shape[2:] == (5,2):

    #      if bboxes.shape[2:] == (5,2):
    #         bboxes = bboxes[:,:,:4]

    return aug_bboxes


def augment_bbox(bbox,transform_mat,return_aug_polygon=False):
    # augment_bbox() can take in bbox OR coords
    #  - for augmenting a bbox, use bbox
    #  - for inverting an augmented bbox, use coords
    
    # augment bbox
    # - - - - - - 
    
    if bbox.shape == (4,):
    # if bbox.shape[2:] == (4,):  
        # augment the bbox
        x1,y1,x2,y2 = bbox
        #[[0,1,0,0],
        # [0,1,0,0],
        # [0,0,0,1],
        # [0,0,0,1]]

        #[[1,0,0,0],
        # [0,0,1,0],
        # [1,0,0,0],
        # [0,0,1,0]]

        # ones = torch.ones(bbox.shape[:2].append(1))

        upper_left  = torch.FloatTensor([x1,y1,1])
        upper_right = torch.FloatTensor([x2,y1,1])
        lower_left  = torch.FloatTensor([x1,y2,1])
        lower_right = torch.FloatTensor([x2,y2,1])



    elif bbox.shape == (5,2) or bbox.shape == (4,2):
        # reverse the bbox's augmentation
        # bbox is [ul, ur, lr, ll, ul] or missing last "ul"
        upper_left  = torch.FloatTensor(np.append(bbox[0],1))
        upper_right = torch.FloatTensor(np.append(bbox[1],1))
        lower_right = torch.FloatTensor(np.append(bbox[2],1))
        lower_left  = torch.FloatTensor(np.append(bbox[3],1))
    else:
        raise Exception("bbox shape is not compatible", bbox.shape)
    
    # print(upper_left, upper_left.shape, bbox[0])
    new_ul = torch.matmul(transform_mat,upper_left)[:2]
    new_ur = torch.matmul(transform_mat,upper_right)[:2]
    new_ll = torch.matmul(transform_mat,lower_left)[:2]
    new_lr = torch.matmul(transform_mat,lower_right)[:2]
    
    coords = [new_ul, new_ur, new_lr, new_ll]
    five_coords = coords + [new_ul] # append for closer loop
    # xs,ys = zip(*five_coords)
    # new_coords = 

    # create surrounding bbox
    # - - - - - - - - - - - -
    torchcoords = torch.stack(five_coords)
    all_x_coords = torchcoords[:,0]
    all_y_coords = torchcoords[:,1]
    
    full_x1 = torch.min(all_x_coords)
    full_x2 = torch.max(all_x_coords)
    full_y1 = torch.min(all_y_coords)
    full_y2 = torch.max(all_y_coords)
    
    aug_bbox = torch.FloatTensor([full_x1, full_y1, full_x2, full_y2])
    
    if return_aug_polygon:
        return aug_bbox,torchcoords
    
    return aug_bbox

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


# class ToTensor(object):
#     def __call__(self, cvimage, boxes=None, labels=None):
#         return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image

class SSDNormalize(object):
    def __init__(self,mean=(0,0,0),std=(1,1,1)):
        self.mean = np.asarray(mean,dtype=np.float32)
        self.std  = np.asarray(std, dtype=np.float32)
        # self.normalize = Normalize(self.mean,self.std)
        # self.toTensor = ToTensor()
    
    def __call__(self, image, boxes, labels):
        # Also can do:
        # image = image-self.mean
        image = (image-self.mean)/self.std
        # image = np.moveaxis(image,[0, 1, 2], [1,2,0])
        # image = self.toTensor(image)
        # image = self.normalize(image)
        # image = image.to_numpy()
        return image, boxes, labels


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)

class RandomLightingNoiseTorch(object):
    def __init__(self):
        self.perms = ([0, 1, 2], [0, 2, 1],
                      [1, 0, 2], [1, 2, 0],
                      [2, 0, 1], [2, 1, 0])

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            image = image[swap,:,:] # shuffle channels, assume (3,H,W)
        return image, boxes, labels

class SSDTorchAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123), std=(1,1,1)):
        self.mean = mean
        self.std  = std
        self.size = size
        self.augment = Compose([
            ToAbsoluteCoords(), # doesn't affect image
            ColorJitter(),
            RandomLightingNoiseTorch(),
            Pad(),
            RandomResizedCrop(),
            RandomHorizontalFlip(),
            SSDNormalize(self.mean,self.std)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)

class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123), std=(1,1,1)):
        self.mean = mean
        self.std  = std
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            # SubtractMeans(self.mean),
            SSDNormalize(self.mean,self.std)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
