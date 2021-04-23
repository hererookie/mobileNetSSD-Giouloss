from ..transforms.transforms import *


class TrainAugmentation:
    # def __init__(self, size, mean=0, std=1.0):
    def __init__(self, size):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = (123.675, 116.28, 103.53)
        self.size = size
        self.std = (58.395, 57.12, 57.375)
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),  # 计算真实的锚点框坐标
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            Standform(self.mean,self.std)
            # lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            # ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            # lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            # ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image