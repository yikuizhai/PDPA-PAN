type='vehicle4'
file_name = "L315_vehicle"
split_size = 640
root = "/home/msi/project/datasets/"
source_dataset_path = root + file_name
target_dataset_path = source_dataset_path + f'_{split_size}_final_ms'


# available labels: train, val, test, trainval
tasks=[
    dict(
        label='train',
        config=dict(
            subimage_size=split_size,
            overlap_size=int(128),
            multi_scale=[1.,],
            horizontal_flip=False,
            vertical_flip=False,
            rotation_angles=[0.]
        )
    ),
    dict(
        label='test',
        config=dict(
            subimage_size=split_size,
            overlap_size=int(128),
            multi_scale=[1.,],
            horizontal_flip=False,
            vertical_flip=False,
            rotation_angles=[0.]
        )
    ),
    dict(
        label='val',
        config=dict(
            subimage_size=split_size,
            overlap_size=int(128),
            multi_scale=[1.,],
            horizontal_flip=False,
            vertical_flip=False,
            rotation_angles=[0.]
        )
   ),
]
