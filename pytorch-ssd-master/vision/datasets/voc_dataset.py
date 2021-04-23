import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os
import torch


class VOCDataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False,
                 label_file=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.root / "ImageSets/Main/test.txt"
        else:
            image_sets_file = self.root / "ImageSets/Main/trainval.txt"
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        # if the labels file exists, read in the class names
        label_file_name = self.root / "labels.txt"

        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            # classes should be a comma separated list

            classes = class_string.split(',')
            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            classes = [elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("VOC Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default VOC classes.")
            self.class_names = ('BACKGROUND',
                                'object', 'bicycle', 'bird', 'boat',
                                'bottle', 'bus', 'car', 'cat', 'chair',
                                'cow', 'diningtable', 'dog', 'horse',
                                'motorbike', 'person', 'pottedplant',
                                'sheep', 'sofa', 'train', 'tvmonitor')
            # self.class_names = ('BACKGROUND',
            #                     'object')

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    # VOCDataset(dataset_path, transform=train_transform, target_transform=target_transform)

    def __getitem__(self, index):
        image_id = self.ids[index]
        # boxes, labels, is_difficult = self._get_annotation(image_id)
        target = self._get_annotation(image_id)
        target = np.array(target)

        boxes = target[:, :4]
        labels = target[:, 4]

        image = self._read_image(image_id)
        # print("*************image_id:\n", image_id)
        # print("bndbox:", boxes)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
            imgs = image[:, :, (2, 1, 0)]  # to rbg
            # print("img**\/\/\/\/\/****shape:", imgs.shape)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        miss = np.array(imgs)
        # print("*************image_id:\n", image_id)
        # print("*************target:\n", target)
        return torch.from_numpy(miss).permute(2, 0, 1), target

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                # print("line.rstrip():", line.rstrip())
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        # boxes = []
        # labels = []
        # is_difficult = []
        res = []
        height = float(ET.parse(annotation_file).find("size").find("height").text)
        # print("**************height**************:", height)
        width = float(ET.parse(annotation_file).find("size").find("width").text)
        # print("**************width**************:", width)
        for obj in objects:
            difficult = int(obj.find('difficult').text) == 1  # 判断该目标是否为难例
            # 判断是否跳过难例
            if not self.keep_difficult and difficult:
                continue
            class_name = obj.find('name').text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = obj.find('bndbox')

                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox = []
                for i, pt in enumerate(pts):
                    cur_pt = int(bbox.find(pt).text) - 1  # 获得坐标值

                    # 将坐标转换成[0,1]，这样图片尺寸发生变化的时候，真实框也随之变化，即平移不变形
                    cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                    bndbox.append(cur_pt)
                # print("bndbox:\\\\\\\\\\\\\\\\\\", bndbox)
                label_idx = self.class_dict[class_name]  # 获得名字对应的label
                bndbox.append(label_idx)

                res.append(bndbox)  # [xmin, ymin, xmax, ymax, label_ind]

                # VOC dataset format follows Matlab, in which indexes start from 0
                # x1 = float(bbox.find('xmin').text) - 1
                # y1 = float(bbox.find('ymin').text) - 1
                # x2 = float(bbox.find('xmax').text) - 1
                # y2 = float(bbox.find('ymax').text) - 1
                # boxes.append([x1, y1, x2, y2])
                #
                # labels.append(self.class_dict[class_name])
                # is_difficult_str = object.find('difficult').text
                # is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)
        # print("res:\n", res)
        return res
        # return (np.array(boxes, dtype=np.float32),
        #         np.array(labels, dtype=np.int64),
        #         np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.root / f"JPEGImages/{image_id}.jpg"
        image = cv2.imread(str(image_file))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    自定义处理在同一个batch,含有不同数量的目标框的情况

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    # imgs = torch.from_numpy(imgs)
    return torch.stack(imgs, 0), targets
