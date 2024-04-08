dataset_info = dict(
    dataset_name='mpii',
    paper_info=dict(
        author='Mykhaylo Andriluka and Leonid Pishchulin and '
        'Peter Gehler and Schiele, Bernt',
        title='2D Human Pose Estimation: New Benchmark and '
        'State of the Art Analysis',
        container='IEEE Conference on Computer Vision and '
        'Pattern Recognition (CVPR)',
        year='2014',
        homepage='http://human-pose.mpi-inf.mpg.de/',
    ),
    # COCO <-> MPII mappings
    keypoint_info={
        16:
        dict(
            name='right_ankle',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        14:
        dict(
            name='right_knee',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        12:
        dict(
            name='right_hip',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        13:
        dict(
            name='left_knee',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        15:
        dict(
            name='left_ankle',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        8:
        dict(
            name='right_elbow',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        6:
        dict(
            name='right_shoulder',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
    },
    skeleton_info={
        0:
        dict(link=('right_ankle', 'right_knee'), id=0, color=[255, 128, 0]),
        1:
        dict(link=('right_knee', 'right_hip'), id=1, color=[255, 128, 0]),
        3:
        dict(link=('right_hip', 'left_hip'), id=3, color=[0, 255, 0]),
        4:
        dict(link=('left_hip', 'left_knee'), id=4, color=[0, 255, 0]),
        5:
        dict(link=('left_knee', 'left_ankle'), id=5, color=[0, 255, 0]),
        6:
        dict(link=('left_hip', 'left_shoulder'), id=5, color=[0, 255, 0]),
        7:
        dict(link=('right_hip', 'right_shoulder'), id=5, color=[0, 255, 0]),
        9:
        dict(link=('left_shoulder', 'right_shoulder'), id=9, color=[51, 153, 255]),
        10:
        dict(
            link=('right_shoulder', 'right_elbow'), id=10, color=[255, 128,
                                                                  0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        13:
        dict(link=('left_shoulder', 'left_elbow'), id=13, color=[0, 255, 0]),
        14:
        dict(link=('left_elbow', 'left_wrist'), id=14, color=[0, 255, 0])
    },
    joint_weights=[
        1.5, 1.2, 1., 1., 1.2, 1.5, 1., 1., 1., 1., 1.5, 1.2, 1., 1., 1.2, 1.5, 1.
    ],
    # Adapted from COCO dataset.
    sigmas=[
        0.089, 0.083, 0.107, 0.107, 0.083, 0.089, 0.026, 0.026, 0.026, 0.026,
        0.062, 0.072, 0.179, 0.179, 0.072, 0.062, 0.6
    ]
    # Directly from COCO to see if sigmas change testing evaluation
    # sigmas=[
    #     0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
    #     0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    # ]
    # This sigmas have no impact on testing
    # maybe has impact on MPII
    )
