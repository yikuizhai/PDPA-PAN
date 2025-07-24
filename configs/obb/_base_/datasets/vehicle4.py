import os
import numpy as np
from tqdm import tqdm

from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.builder import DATASETS
from mmdet.core import eval_map, poly2obb
from BboxToolkit.evaluate.dota_eval import voc_eval_dota
from BboxToolkit.transforms import bbox2type


@DATASETS.register_module()
class Vehicle4Dataset(CustomDataset):
    CLASSES = ('car', 'truck', 'bus', 'tank')  # 根据vehicle4实际类别替换

    def __init__(self, **kwargs):
        super(Vehicle4Dataset, self).__init__(**kwargs)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        
        # 转换标注格式为OBBDetection所需格式
        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, x2, y2, x3, y3, x4, y4 = ann['bbox']
            polygon = np.array([x1, y1, x2, y2, x3, y3, x4, y4])
            bbox = poly2obb(polygon)  # 多边形转换为OBB格式
            
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 5), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 5), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann)

        return ann

    def format_results(self, results, save_dir=None, **kwargs):
        """Format results for DOTA evaluation."""
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            
        det_results = []
        for idx, result in enumerate(results):
            img_name = os.path.splitext(os.path.basename(self.img_infos[idx]['filename']))[0]
            
            for label, dets in enumerate(result):
                if len(dets) == 0:
                    continue
                    
                # 转换检测结果为多边形格式
                polys = bbox2type(dets[:, :5], 'obb', 'poly')
                scores = dets[:, -1]
                
                for poly, score in zip(polys, scores):
                    det_results.append({
                        'img_name': img_name,
                        'cls_name': self.CLASSES[label],
                        'poly': poly,
                        'score': score
                    })
                    
                    if save_dir:
                        with open(os.path.join(save_dir, f"{self.CLASSES[label]}.txt"), 'a') as f:
                            f.write(f"{img_name} {score:.4f} {' '.join([str(p) for p in poly])}\n")
                            
        return det_results

    def evaluate(self, results, metric='mAP', logger=None, **kwargs):
        """Evaluate the dataset using DOTA evaluation metrics."""
        if not isinstance(results, list):
            raise TypeError('results must be a list')
            
        # 将结果保存到临时目录并评估
        import tempfile
        tmp_dir = tempfile.TemporaryDirectory()
        det_results = self.format_results(results, save_dir=tmp_dir.name)
        
        # 获取GT信息
        gt_info = {}
        for idx in range(len(self.img_infos)):
            img_id = self.img_infos[idx]['id']
            ann_info = self.get_ann_info(idx)
            gt_info[img_id] = ann_info
            
        # 使用BboxToolkit的评估函数
        eval_results = {}
        mean_ap = 0.0
        for i, cls_name in enumerate(self.CLASSES):
            result_file = os.path.join(tmp_dir.name, f"{cls_name}.txt")
            if not os.path.exists(result_file):
                eval_results[cls_name + '_AP'] = 0.0
                continue
                
            rec, prec, ap = voc_eval_dota(result_file, gt_info, cls_name)
            eval_results[cls_name + '_AP'] = ap
            mean_ap += ap
            
        eval_results['mAP'] = mean_ap / len(self.CLASSES)
        tmp_dir.cleanup()
        
        return eval_results