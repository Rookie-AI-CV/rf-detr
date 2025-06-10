import cv2
import numpy as np
import torch
import torchvision.transforms.functional as VF
from PIL import Image
import random
from . import box_utils
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def ensure_cv2(img):
    """
    确保图像为OpenCV格式
    Args:
        img: 输入图像，可以是PIL Image或numpy数组
    Returns:
        OpenCV格式的图像
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pass
    elif isinstance(img, np.ndarray):
        pass
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
    
    return img

def letterbox_torch(img, bboxes, new_shape=(640, 640), fill=(0.44, 0.44, 0.44)):
    """
    使用PyTorch实现的letterbox变换
    Args:
        img: 输入图像 [C, H, W]
        bboxes: 边界框坐标
        new_shape: 目标尺寸 (H, W)
        fill: 填充颜色
    Returns:
        变换后的图像和边界框
    """
    # new_shape: [H, W]
    # img: [C, H, W]
    # Scale ratio (new / old)
    shape = img.shape[1:3]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)  # only scale down, do not scale up (for better val mAP)

    # Compute padding
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]  # wh padding

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = VF.resize(img, new_unpad)
        pass

    top = int(round(dh - 0.1))
    left = int(round(dw - 0.1))
    img_ = torch.ones((3, new_shape[0], new_shape[1]), dtype=img.dtype)
    img_[:, top : top + new_unpad[0], left : left + new_unpad[1]] = img

    boxes_ = bboxes.copy()
    boxes_[:, 0] = boxes_[:, 0] * r + left
    boxes_[:, 1] = boxes_[:, 1] * r + top
    boxes_[:, 2] = boxes_[:, 2] * r + left
    boxes_[:, 3] = boxes_[:, 3] * r + top
    return img_, boxes_

    pass


def copy_boxes(boxes):
    """
    复制边界框数据
    Args:
        boxes: 输入边界框，可以是tensor或numpy数组
    Returns:
        边界框的深拷贝
    """
    if isinstance(boxes, torch.Tensor):
        return boxes.clone()
    return boxes.copy()


class Letterbox:
    """
    图像letterbox变换类
    保持图像宽高比，在较短的边填充像素
    """
    def __init__(self, new_shape=(640, 640)):
        self.new_shape = new_shape
        self.scaleup = False  # 是否允许放大
        self.auto = False     # 是否自动计算填充
        self.scale_fill = False  # 是否拉伸填充
        self.center = True    # 是否居中填充
        pass

    def __call__(self, img, target):
        shape = img.shape[:2]
        new_shape = self.new_shape

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        
        # update boxes
        if 'boxes' in target:
            boxes = copy_boxes(target['boxes'])
            boxes[:, 0] = boxes[:, 0] * ratio[0] + left
            boxes[:, 1] = boxes[:, 1] * ratio[1] + top
            boxes[:, 2] = boxes[:, 2] * ratio[0] + left
            boxes[:, 3] = boxes[:, 3] * ratio[1] + top
            boxes[:, 0] = np.clip(boxes[:, 0], 0, new_shape[1])
            boxes[:, 1] = np.clip(boxes[:, 1], 0, new_shape[0])
            boxes[:, 2] = np.clip(boxes[:, 2], 0, new_shape[1])
            boxes[:, 3] = np.clip(boxes[:, 3], 0, new_shape[0])

            target['boxes'] = boxes

        return img, target


class RandomHorizontalFlip:
    """
    随机水平翻转增强
    以概率p对图像和边界框进行水平翻转
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() <= self.p and 'boxes' in target:
            img = cv2.flip(img, 1)
            bboxes_ = copy_boxes(target['boxes'])
            bboxes_[:, 0] = img.shape[1] - target['boxes'][:, 2]
            bboxes_[:, 2] = img.shape[1] - target['boxes'][:, 0]
            bboxes_[:, 0] = np.clip(bboxes_[:, 0], 0, img.shape[1])
            bboxes_[:, 2] = np.clip(bboxes_[:, 2], 0, img.shape[1])

            target['boxes'] = bboxes_

            return img, target
        return img, target


class RandomVerticalFlip:
    """
    随机垂直翻转增强
    以概率p对图像和边界框进行垂直翻转
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() <= self.p and 'boxes' in target:
            img = cv2.flip(img, 0)
            bboxes_ = copy_boxes(target['boxes'])
            bboxes_[:, 1] = img.shape[0] - target['boxes'][:, 3]
            bboxes_[:, 3] = img.shape[0] - target['boxes'][:, 1]
            bboxes_[:, 1] = np.clip(bboxes_[:, 1], 0, img.shape[0])
            bboxes_[:, 3] = np.clip(bboxes_[:, 3], 0, img.shape[0])

            target['boxes'] = bboxes_

            return img, target
        return img, target
    pass


class RandomRotate90:
    """
    随机90度旋转增强
    以概率p对图像和边界框进行90度旋转
    """
    def __init__(self, p=0.5):
        self.p = p
        self.transform = iaa.Rot90((1, 3))

    def __call__(self, img, target):
        if random.random() <= self.p and 'boxes' in target:
            bboxes = target['boxes']
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.cpu().numpy()

            bbs = BoundingBoxesOnImage([
                BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
                for box in bboxes
            ], shape=img.shape)
            image_aug, bbs_aug = self.transform(image=img, bounding_boxes=bbs)
            bboxes_ = np.zeros_like(bboxes)
            for i, box in enumerate(bbs_aug.bounding_boxes):
                bboxes_[i, 0] = box.x1
                bboxes_[i, 1] = box.y1
                bboxes_[i, 2] = box.x2
                bboxes_[i, 3] = box.y2

            img = image_aug
            target['boxes'] = torch.from_numpy(bboxes_).to(target['boxes'].device) if isinstance(target['boxes'], torch.Tensor) else bboxes_

            return img, target
        return img, target
    pass


class RandomRotate:
    """
    随机旋转增强
    以概率p对图像和边界框进行-10到10度的随机旋转
    """
    def __init__(self, p=0.5):
        self.p = p
        self.transform = iaa.Affine(rotate=(-10, 10))

    def __call__(self, img, target):
        if random.random() <= self.p and 'boxes' in target:
            bboxes = target['boxes']
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.cpu().numpy()

            bbs = BoundingBoxesOnImage([
                BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
                for box in bboxes
            ], shape=img.shape)

            image_aug, bbs_aug = self.transform(image=img, bounding_boxes=bbs)    

            bboxes_ = np.zeros_like(bboxes)
            for i, box in enumerate(bbs_aug.bounding_boxes):
                bboxes_[i, 0] = box.x1
                bboxes_[i, 1] = box.y1
                bboxes_[i, 2] = box.x2
                bboxes_[i, 3] = box.y2

            img = image_aug
            target['boxes'] = torch.from_numpy(bboxes_).to(target['boxes'].device) if isinstance(target['boxes'], torch.Tensor) else bboxes_

            return img, target
        return img, target
    pass


class RandomAffine:
    """
    随机仿射变换增强
    以概率p对图像和边界框进行-10到10度的随机剪切变换
    """
    def __init__(self, p=0.5):
        self.p = p
        self.transform = iaa.Affine(shear=(-10, 10))

    def __call__(self, img, target):
        if random.random() <= self.p and 'boxes' in target:
            bboxes = target['boxes']
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.cpu().numpy()

            bbs = BoundingBoxesOnImage([
                BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
                for box in bboxes
            ], shape=img.shape)
            image_aug, bbs_aug = self.transform(image=img, bounding_boxes=bbs)    

            bboxes_ = np.zeros_like(bboxes)
            for i, box in enumerate(bbs_aug.bounding_boxes):
                bboxes_[i, 0] = box.x1
                bboxes_[i, 1] = box.y1
                bboxes_[i, 2] = box.x2
                bboxes_[i, 3] = box.y2

            img = image_aug
            target['boxes'] = torch.from_numpy(bboxes_).to(target['boxes'].device) if isinstance(target['boxes'], torch.Tensor) else bboxes_

            return img, target
        return img, target
    pass


class RandomCrop:
    """
    随机裁剪增强
    以概率p对图像进行随机裁剪，同时调整边界框位置
    """
    def __init__(self, p=0.5, min_size=0.5):
        self.p = p
        self.min_size = min_size

    def __call__(self, img, target):
        if random.random() < self.p and 'boxes' in target:
            h, w = img.shape[:2]
            
            # choose x
            xrange = int(w * (1 - self.min_size))
            x1 = random.randint(0, xrange)
            x2 = random.randint(x1 + int(w * self.min_size), w)
            # choose y
            yrange = int(h * (1 - self.min_size))
            y1 = random.randint(0, yrange)
            y2 = random.randint(y1 + int(h * self.min_size), h)

            # crop
            img = img[y1:y2, x1:x2].copy()

            # remove boxes outside the crop
            boxes = target['boxes']
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            mask = (boxes[:, 0] < x2) & (boxes[:, 2] > x1) & (boxes[:, 1] < y2) & (boxes[:, 3] > y1)
            boxes = boxes[mask]
            if 'labels' in target:
                target['labels'] = target['labels'][mask]
            if 'area' in target:
                target['area'] = target['area'][mask]

            # adjust boxes
            boxes_ = boxes.copy()
            boxes_[:, 0] = np.clip(boxes_[:, 0] - x1, 0, img.shape[1])
            boxes_[:, 1] = np.clip(boxes_[:, 1] - y1, 0, img.shape[0])
            boxes_[:, 2] = np.clip(boxes_[:, 2] - x1, 0, img.shape[1])
            boxes_[:, 3] = np.clip(boxes_[:, 3] - y1, 0, img.shape[0])
            target['boxes'] = torch.from_numpy(boxes_).to(target['boxes'].device) if isinstance(target['boxes'], torch.Tensor) else boxes_

            return img, target
        return img, target
    pass


class RandomNoise:
    """
    随机噪声增强
    以概率p向图像添加高斯噪声
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() <= self.p:
            img = img.astype(np.float32)
            noise = np.random.normal(0, 0.02, img.shape) * 255
            img = img + noise
            img = np.clip(img, 0, 255).astype(np.uint8)
            return img, target
        return img, target
    pass


class RandomBlur:
    """
    随机模糊增强
    以概率p对图像进行随机高斯模糊
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if isinstance(img, Image.Image):
            img = np.array(img)
            
        if np.random.rand() < self.p:
            ksize = np.random.randint(3, 7, 2) * 2 + 1
            img = cv2.GaussianBlur(img, ksize, 0)
            return img, target
        return img, target
    pass


class RandomBrightness:
    """
    随机亮度调整增强
    以概率p随机调整图像亮度
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() <= self.p:
            alpha = np.random.uniform(0.9, 1.1)
            img = cv2.convertScaleAbs(img, alpha=alpha)
            return img, target
        return img, target
    pass


class RandomGrayScale:
    """
    随机灰度转换增强
    以概率p将图像转换为灰度图
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() <= self.p:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img, target
        return img, target
    pass


class RandomResize:
    """
    随机缩放增强
    以概率p对图像和边界框进行随机缩放
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() <= self.p and 'boxes' in target:
            scale = np.random.uniform(0.5, 1.5)
            new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            target['boxes'] = target['boxes'] * scale
            return img, target
        return img, target
    pass


class RandomPerspective:
    """
    随机透视变换增强
    以概率p对图像和边界框进行随机透视变换
    """
    def __init__(self, p=0.5):
        self.p = p
        self.transform = iaa.PerspectiveTransform(scale=(0.0, 0.1))

    def __call__(self, img, target):
        if random.random() <= self.p and 'boxes' in target:
            bboxes = target['boxes']
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.cpu().numpy()

            bbs = BoundingBoxesOnImage([
                BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
                for box in bboxes
            ], shape=img.shape)

            image_aug, bbs_aug = self.transform(image=img, bounding_boxes=bbs)

            bboxes_ = np.zeros_like(bboxes)
            for i, box in enumerate(bbs_aug.bounding_boxes):
                bboxes_[i, 0] = box.x1
                bboxes_[i, 1] = box.y1
                bboxes_[i, 2] = box.x2
                bboxes_[i, 3] = box.y2

            img = image_aug
            target['boxes'] = torch.from_numpy(bboxes_).to(target['boxes'].device) if isinstance(target['boxes'], torch.Tensor) else bboxes_

            return img, target
        return img, target
    pass


class FilterSmallBox:
    """
    过滤小目标框
    移除尺寸小于阈值的边界框
    """
    def __init__(self, min_size=3):
        self.min_size = min_size  # 最小框尺寸阈值

    def __call__(self, img, target):
        if 'boxes' in target:
            boxes = target['boxes']
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()  # 将tensor转换为numpy数组
            width = boxes[:, 2] - boxes[:, 0]  # 计算框的宽度
            height = boxes[:, 3] - boxes[:, 1]  # 计算框的高度
            mask = (width >= self.min_size) & (height >= self.min_size)  # 创建掩码过滤小框
            target['boxes'] = boxes[mask]  # 更新过滤后的框
            if 'labels' in target:
                target['labels'] = target['labels'][mask]  # 更新对应的标签
            if 'area' in target:
                target['area'] = target['area'][mask]  # 更新对应的面积
            return img, target
        return img, target
    pass


class RandomShuffleChannel:
    """
    随机通道打乱增强
    以概率p随机打乱图像RGB通道顺序
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() <= self.p:
            channels = [0, 1, 2]  # RGB三个通道
            np.random.shuffle(channels)  # 随机打乱通道顺序
            img = img[:, :, channels]  # 按照新的通道顺序重组图像
            return img, target
        return img, target
    pass


class ToNumpy:
    """
    转换为NumPy数组
    将输入图像转换为OpenCV格式的NumPy数组
    """
    def __call__(self, img, target):
        img = ensure_cv2(img)
        return img, target
    pass


class ToTensor:
    """
    转换为PyTorch张量
    将图像转换为PyTorch张量格式
    """
    def __call__(self, img, target):
        img = VF.to_tensor(img)
        return img, target
    pass


class BGR2RGB:
    """
    BGR转RGB
    将OpenCV的BGR格式转换为RGB格式
    """
    def __call__(self, img, target):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, target
    pass


class Resize:
    """
    图像缩放
    将图像缩放到指定最大尺寸，保持宽高比
    参数:
        max_size: 图像的最大边长，默认为640
    功能:
        1. 计算图像的宽高比
        2. 如果图像最大边长超过max_size，则按比例缩放
        3. 同时缩放目标框坐标
        4. 保持图像宽高比不变
    """
    def __init__(self, max_size=640):
        self.max_size = max_size

    def __call__(self, img, target):
        # 获取图像高度和宽度
        h, w = img.shape[:2]
        # 计算最大边长
        max_hw = max(h, w)
        
        # 如果最大边长超过限制，进行缩放
        if max_hw > self.max_size:
            # 计算缩放比例
            scale = self.max_size / max_hw
            # 计算新的高度和宽度
            new_h, new_w = int(h * scale), int(w * scale)
            # 使用双线性插值进行图像缩放
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 如果存在目标框，同步缩放目标框坐标
            if 'boxes' in target:
                target['boxes'] = target['boxes'] * scale
        else:
            # 如果图像尺寸小于限制，保持原尺寸
            new_h, new_w = h, w

        return img, target


class Compose:
    """
    组合多个变换
    按顺序应用多个数据增强变换
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target
    
    def extend(self, transforms):
        if isinstance(transforms, list):
            self.transforms.extend(transforms)
        else:
            raise TypeError(f"Unsupported type: {type(transforms)}")
        pass


class Format:
    """
    格式化输出
    将边界框转换为标准格式，包括xyxy和归一化的cxcywh格式
    """
    def __init__(self, input_size=(640, 640)):
        self.input_size = input_size

    def __call__(self, img, target):
        if 'boxes' in target:
            boxes = target['boxes']
            # Convert boxes to tensor if it's numpy array
            if isinstance(boxes, np.ndarray):
                boxes = torch.from_numpy(boxes).float()
            width, height = img.shape[1], img.shape[0]

            # convert xyxy to cxcywh
            bboxes_xyxy = boxes.clone()
            bboxes_cxcywh_norm = box_utils.normalize(box_utils.xyxy2cxcywh(bboxes_xyxy), height, width)
            
            target['bboxes_xyxy'] = bboxes_xyxy
            target['bboxes_cxcywh_norm'] = bboxes_cxcywh_norm
            target['boxes'] = bboxes_xyxy
            target['batch_idx'] = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        if 'labels' in target:
            if isinstance(target['labels'], np.ndarray):
                target['labels'] = torch.from_numpy(target['labels']).long()
            else:
                target['labels'] = target['labels'].long()

        return img, target

