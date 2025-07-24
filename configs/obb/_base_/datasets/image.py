import os
import numpy as np
from PIL import Image

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


@DATASETS.register_module()
class ImageDataset(CustomDataset):
    """Dataset for test without annotations."""
    
    def __init__(self, 
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_file=None,
                 test_mode=True,
                 filter_empty_gt=False,
                 classes=None):
        
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.test_mode = test_mode
        
        # 载入图片列表
        if ann_file:
            # 如果提供了注释文件，使用它作为图像列表
            with open(ann_file, 'r') as f:
                self.img_infos = []
                for line in f:
                    img_name = line.strip()
                    self.img_infos.append({'filename': img_name})
        else:
            # 否则，扫描图像目录
            self.img_infos = []
            for img_name in os.listdir(img_dir):
                if self._is_image_file(img_name):
                    self.img_infos.append({'filename': img_name})
        
        # 继承CustomDataset，但跳过标注加载
        super(ImageDataset, self).__init__(
            ann_file=None,
            pipeline=pipeline,
            img_prefix=img_dir,
            test_mode=True,
            filter_empty_gt=False,
            classes=classes)

    def _is_image_file(self, filename):
        """Check if a file is an image."""
        return filename.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
            
    def get_ann_info(self, idx):
        """Get annotation by index.
        For test mode, return empty annotations.
        """
        return {'bboxes': np.zeros((0, 5), dtype=np.float32),
                'labels': np.zeros((0,), dtype=np.int64)}
    
    def prepare_test_img(self, idx):
        """Prepare an image for testing."""
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)