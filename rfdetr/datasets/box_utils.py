import numpy as np
import torch

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def _to_tensor(x, device=None):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return x

def xyxy2xywh(boxes):
    """
    Convert bounding boxes from (x1, y1, x2, y2) format to (x, y, w, h) format.
    Args:
        boxes (np.ndarray or torch.Tensor): Bounding boxes in (x1, y1, x2, y2) format.
    Returns:
        np.ndarray or torch.Tensor: Bounding boxes in (x, y, w, h) format.
    """
    is_tensor = isinstance(boxes, torch.Tensor)
    device = boxes.device if is_tensor else None
    boxes = _to_numpy(boxes)
    
    boxes_ = boxes.copy()
    if boxes_.ndim == 1:
        boxes_ = boxes_[None, :]
        squeeze = True
    else:
        squeeze = False

    x = boxes_[:, 0]
    y = boxes_[:, 1]
    w = boxes_[:, 2] - boxes_[:, 0]
    h = boxes_[:, 3] - boxes_[:, 1]
    boxes_ = np.stack((x, y, w, h), axis=1)
    if squeeze:
        boxes_ = boxes_[0, :]

    return _to_tensor(boxes_, device) if is_tensor else boxes_


def xyxy2cxcywh(boxes):
    """
    Convert bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h) format.
    Args:
        boxes (np.ndarray or torch.Tensor): Bounding boxes in (x1, y1, x2, y2) format.
    Returns:
        np.ndarray or torch.Tensor: Bounding boxes in (cx, cy, w, h) format.
    """
    is_tensor = isinstance(boxes, torch.Tensor)
    device = boxes.device if is_tensor else None
    boxes = _to_numpy(boxes)
    
    boxes_ = boxes.copy()
    if boxes_.ndim == 1:
        boxes_ = boxes_[None, :]
        squeeze = True
    else:
        squeeze = False

    x = boxes_[:, 0]
    y = boxes_[:, 1]
    w = boxes_[:, 2] - boxes_[:, 0]
    h = boxes_[:, 3] - boxes_[:, 1]
    cx = x + w / 2
    cy = y + h / 2

    boxes_ = np.stack((cx, cy, w, h), axis=1)
    if squeeze:
        boxes_ = boxes_[0, :]

    return _to_tensor(boxes_, device) if is_tensor else boxes_


def normalize(boxes, height, width):
    """
    Normalize bounding boxes to [0, 1] range.
    Args:
        boxes (np.ndarray or torch.Tensor): Bounding boxes in (cx, cy, w, h) format.
        height (int): Image height.
        width (int): Image width.
    Returns:
        np.ndarray or torch.Tensor: Normalized bounding boxes.
    """
    is_tensor = isinstance(boxes, torch.Tensor)
    device = boxes.device if is_tensor else None
    boxes = _to_numpy(boxes)
    
    boxes_ = boxes.copy()
    if boxes_.ndim == 1:
        boxes_ = boxes_[None, :]
        squeeze = True
    else:
        squeeze = False

    boxes_[:, 0] = boxes_[:, 0] / width
    boxes_[:, 1] = boxes_[:, 1] / height
    boxes_[:, 2] = boxes_[:, 2] / width
    boxes_[:, 3] = boxes_[:, 3] / height

    boxes_ = np.clip(boxes_, 0, 1)

    boxes_[:, 2] = np.where(boxes_[:, 0] - boxes_[:, 2] * 0.5 < 0, boxes_[:, 0]*0.5, boxes_[:, 2])
    boxes_[:, 2] = np.where(boxes_[:, 0] + boxes_[:, 2] * 0.5 > 1, (1 - boxes_[:, 0]) * 0.5, boxes_[:, 2])
    boxes_[:, 3] = np.where(boxes_[:, 1] - boxes_[:, 3] * 0.5 < 0, boxes_[:, 1]*0.5, boxes_[:, 3])
    boxes_[:, 3] = np.where(boxes_[:, 1] + boxes_[:, 3] * 0.5 > 1, (1 - boxes_[:, 1]) * 0.5, boxes_[:, 3])

    if squeeze:
        boxes_ = boxes_[0, :]

    return _to_tensor(boxes_, device) if is_tensor else boxes_


def cxcywh2xyxy(boxes):
    """
    Convert bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    Args:
        boxes (np.ndarray or torch.Tensor): Bounding boxes in (cx, cy, w, h) format.
    Returns:
        np.ndarray or torch.Tensor: Bounding boxes in (x1, y1, x2, y2) format.
    """
    is_tensor = isinstance(boxes, torch.Tensor)
    device = boxes.device if is_tensor else None
    boxes = _to_numpy(boxes)
    
    if boxes.ndim == 1:
        result = np.array([boxes[0] - boxes[2] / 2, boxes[1] - boxes[3] / 2,
                          boxes[0] + boxes[2] / 2, boxes[1] + boxes[3] / 2], dtype=np.float32)
        return _to_tensor(result, device) if is_tensor else result

    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    result = np.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), axis=1)
    return _to_tensor(result, device) if is_tensor else result


def unnormalize(boxes, width, height):
    """
    Unnormalize bounding boxes from [0, 1] range to pixel coordinates.
    Args:
        boxes (np.ndarray or torch.Tensor): Normalized bounding boxes.
        width (int): Image width.
        height (int): Image height.
    Returns:
        np.ndarray or torch.Tensor: Unnormalized bounding boxes.
    """
    is_tensor = isinstance(boxes, torch.Tensor)
    device = boxes.device if is_tensor else None
    boxes = _to_numpy(boxes)
    
    boxes_ = boxes.copy()
    if boxes_.ndim == 1:
        boxes_ = boxes_[None, :]
        squeeze = True
    else:
        squeeze = False

    boxes_[:, 0] = boxes_[:, 0] * width
    boxes_[:, 1] = boxes_[:, 1] * height
    boxes_[:, 2] = boxes_[:, 2] * width
    boxes_[:, 3] = boxes_[:, 3] * height

    if squeeze:
        boxes_ = boxes_[0, :]

    return _to_tensor(boxes_, device) if is_tensor else boxes_


def iou_xyxy(box1, box2):
    """
    Calculate IoU between two bounding boxes.
    Args:
        box1 (np.ndarray or torch.Tensor): First bounding box in (x1, y1, x2, y2) format.
        box2 (np.ndarray or torch.Tensor): Second bounding box(es) in (x1, y1, x2, y2) format.
    Returns:
        np.ndarray or torch.Tensor: IoU values.
    """
    is_tensor = isinstance(box1, torch.Tensor) or isinstance(box2, torch.Tensor)
    device = box1.device if isinstance(box1, torch.Tensor) else (box2.device if isinstance(box2, torch.Tensor) else None)
    box1 = _to_numpy(box1)
    box2 = _to_numpy(box2)
    
    x1 = np.maximum(box1[0], box2[:, 0])
    y1 = np.maximum(box1[1], box2[:, 1])
    x2 = np.minimum(box1[2], box2[:, 2])
    y2 = np.minimum(box1[3], box2[:, 3])
    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)
    intersection = w * h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - intersection
    iou = intersection / union
    return _to_tensor(iou, device) if is_tensor else iou


def nms(boxes, cls, scores):
    """
    Non-maximum suppression.
    Args:
        boxes (np.ndarray or torch.Tensor): Bounding boxes in (x1, y1, x2, y2) format.
        cls (np.ndarray or torch.Tensor): Class labels.
        scores (np.ndarray or torch.Tensor): Confidence scores.
    Returns:
        list: Indices of kept boxes.
    """
    is_tensor = isinstance(boxes, torch.Tensor) or isinstance(cls, torch.Tensor) or isinstance(scores, torch.Tensor)
    boxes = _to_numpy(boxes)
    cls = _to_numpy(cls)
    scores = _to_numpy(scores)

    # first, sort by scores
    indices = np.argsort(scores)[::-1]
    boxes = boxes[indices]
    cls = cls[indices]
    scores = scores[indices]
    keep = []

    iou_thr_different_cls = 0.6
    iou_thr_same_cls = 0.5
    while len(boxes) > 0:
        # get the first box
        box = boxes[0]
        keep.append(indices[0])

        # calculate the iou of the first box with the rest
        iou = iou_xyxy(box, boxes[1:])

        mask = (iou > iou_thr_same_cls) & (cls[0] == cls[1:])
        mask = mask | ((iou > iou_thr_different_cls) & (cls[0] != cls[1:]))

        mask = np.logical_not(mask)

        if mask.sum() == 0:
            # no more boxes to keep
            break

        boxes = boxes[1:][mask]
        cls = cls[1:][mask]
        scores = scores[1:][mask]
        indices = indices[1:][mask]

    return keep


    


