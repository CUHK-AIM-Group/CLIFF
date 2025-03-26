import json
import torch
from detectron2.structures import Boxes, Instances
from typing import List
import numpy as np


def get_coco_instances(image_ids: List[int], json_path: str) -> List[Instances]:
    """
    根据一组 image_ids 和 JSON 文件路径提取 COCO 数据并生成 Instances 对象列表。
    每个 Instances 包含 GT 边界框、objectness_logits 和 gt_categories。

    Args:
        image_ids (List[int]): COCO 数据集中的图像 ID 列表。
        json_path (str): COCO 验证集 JSON 文件的路径，例如 "path/to/instances_val2017.json"。

    Returns:
        List[Instances]: 包含每个图像的 Instances 对象的列表。
                         每个 Instances 包含 proposal_boxes、objectness_logits 和 gt_categories。

    Raises:
        ValueError: 如果某个 image_id 不存在或没有对应的标注。
    """
    # 加载 COCO JSON 文件
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # 创建图像信息字典
    image_info_dict = {img['id']: {'height': img['height'], 'width': img['width']}
                       for img in coco_data['images']}

    # 创建标注字典，按 image_id 分组，包含边界框和类别
    annotations_dict = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_dict:
            annotations_dict[img_id] = {'boxes': [], 'categories': []}
        # COCO 的 bbox 格式为 [x, y, width, height]，转换为 [x1, y1, x2, y2]
        x, y, w, h = ann['bbox']
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        annotations_dict[img_id]['boxes'].append([x1, y1, x2, y2])
        annotations_dict[img_id]['categories'].append(ann['category_id'])

    # 验证输入并生成 Instances 列表
    instances_list = []
    for image_id in image_ids:
        if image_id not in image_info_dict:
            raise ValueError(f"Image ID {image_id} not found in the dataset.")
        if image_id not in annotations_dict:
            raise ValueError(f"No annotations found for Image ID {image_id}.")

        image_height = image_info_dict[image_id]['height']
        image_width = image_info_dict[image_id]['width']
        gt_boxes = torch.tensor(annotations_dict[image_id]['boxes'], device='cuda:0')
        gt_categories = torch.tensor(annotations_dict[image_id]['categories'], device='cuda:0')

        # 创建 Instances 对象
        proposal_boxes = Boxes(gt_boxes)
        num_instances = len(gt_boxes)
        objectness_logits = torch.full((num_instances,), 10.0, device='cuda:0')

        instances = Instances(image_size=(image_height, image_width))
        instances.proposal_boxes = proposal_boxes
        instances.objectness_logits = objectness_logits
        instances.gt_categories = gt_categories  # 新增 gt_categories

        instances_list.append(instances)

    return instances_list
