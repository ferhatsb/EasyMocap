import numpy as np
import cv2
from easymocap.mytools import Timer
import numpy as np

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def bbox_from_keypoints(keypoints, rescale=1.2, detection_thresh=0.05, MIN_PIXEL=5):
    """Get center and scale for bounding box from openpose detections."""
    valid = keypoints[:, -1] > detection_thresh
    if valid.sum() < 3:
        return [0, 0, 100, 100, 0]
    valid_keypoints = keypoints[valid][:, :-1]
    center = (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0)) / 2
    bbox_size = valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)
    # adjust bounding box tightness
    if bbox_size[0] < MIN_PIXEL or bbox_size[1] < MIN_PIXEL:
        return [0, 0, 100, 100, 0]
    bbox_size = bbox_size * rescale
    bbox = [
        center[0] - bbox_size[0] / 2,
        center[1] - bbox_size[1] / 2,
        center[0] + bbox_size[0] / 2,
        center[1] + bbox_size[1] / 2,
        keypoints[valid, 2].mean()
    ]
    return bbox


class Detector:
    NUM_BODY = 23
    NUM_HAND = 21
    NUM_FACE = 68

    flip_map = [[0, 'nose'], [2, 'right_eye'], [1, 'left_eye'], [4, 'right_ear'], [3, 'left_ear'],
                [6, 'right_shoulder'],
                [5, 'left_shoulder'], [8, 'right_elbow'], [7, 'left_elbow'], [10, 'right_wrist'], [9, 'left_wrist'],
                [12, 'right_hip'], [11, 'left_hip'], [14, 'right_knee'], [13, 'left_knee'], [16, 'right_ankle'],
                [15, 'left_ankle'], [20, 'right_big_toe'], [21, 'right_small_toe'], [22, 'right_heel'],
                [17, 'left_big_toe'],
                [18, 'left_small_toe'], [19, 'left_heel'], [39, 'face-16'], [38, 'face-15'], [37, 'face-14'],
                [36, 'face-13'],
                [35, 'face-12'], [34, 'face-11'], [33, 'face-10'], [32, 'face-9'], [31, 'face-8'], [30, 'face-7'],
                [29, 'face-6'],
                [28, 'face-5'], [27, 'face-4'], [26, 'face-3'], [25, 'face-2'], [24, 'face-1'], [23, 'face-0'],
                [49, 'face-26'],
                [48, 'face-25'], [47, 'face-24'], [46, 'face-23'], [45, 'face-22'], [44, 'face-21'], [43, 'face-20'],
                [42, 'face-19'], [41, 'face-18'], [40, 'face-17'], [50, 'face-27'], [51, 'face-28'], [52, 'face-29'],
                [53, 'face-30'], [58, 'face-35'], [57, 'face-34'], [56, 'face-33'], [55, 'face-32'], [54, 'face-31'],
                [68, 'face-45'], [67, 'face-44'], [66, 'face-43'], [65, 'face-42'], [70, 'face-47'], [69, 'face-46'],
                [62, 'face-39'], [61, 'face-38'], [60, 'face-37'], [59, 'face-36'], [64, 'face-41'], [63, 'face-40'],
                [77, 'face-54'], [76, 'face-53'], [75, 'face-52'], [74, 'face-51'], [73, 'face-50'], [72, 'face-49'],
                [71, 'face-48'], [82, 'face-59'], [81, 'face-58'], [80, 'face-57'], [79, 'face-56'], [78, 'face-55'],
                [87, 'face-64'], [86, 'face-63'], [85, 'face-62'], [84, 'face-61'], [83, 'face-60'], [90, 'face-67'],
                [89, 'face-66'], [88, 'face-65'], [112, 'right_hand_root'], [113, 'right_thumb1'],
                [114, 'right_thumb2'],
                [115, 'right_thumb3'], [116, 'right_thumb4'], [117, 'right_forefinger1'], [118, 'right_forefinger2'],
                [119, 'right_forefinger3'], [120, 'right_forefinger4'], [121, 'right_middle_finger1'],
                [122, 'right_middle_finger2'], [123, 'right_middle_finger3'], [124, 'right_middle_finger4'],
                [125, 'right_ring_finger1'], [126, 'right_ring_finger2'], [127, 'right_ring_finger3'],
                [128, 'right_ring_finger4'], [129, 'right_pinky_finger1'], [130, 'right_pinky_finger2'],
                [131, 'right_pinky_finger3'], [132, 'right_pinky_finger4'], [91, 'left_hand_root'], [92, 'left_thumb1'],
                [93, 'left_thumb2'], [94, 'left_thumb3'], [95, 'left_thumb4'], [96, 'left_forefinger1'],
                [97, 'left_forefinger2'],
                [98, 'left_forefinger3'], [99, 'left_forefinger4'], [100, 'left_middle_finger1'],
                [101, 'left_middle_finger2'],
                [102, 'left_middle_finger3'], [103, 'left_middle_finger4'], [104, 'left_ring_finger1'],
                [105, 'left_ring_finger2'], [106, 'left_ring_finger3'], [107, 'left_ring_finger4'],
                [108, 'left_pinky_finger1'],
                [109, 'left_pinky_finger2'], [110, 'left_pinky_finger3'], [111, 'left_pinky_finger4']]

    def __init__(self, nViews, show=False, **cfg) -> None:
        self.nViews = nViews
        self.show = show
        self.NUM_BODY = 25
        self.openpose25_in_23 = [0, 0, 6, 8, 10, 5, 7, 9, 0, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 17, 18, 19, 20, 21, 22]

        det_config = cfg['det_config']
        det_checkpoint = cfg['det_checkpoint']
        device = cfg['device']
        pose_config = cfg['pose_config']
        pose_checkpoint = cfg['pose_checkpoint']

        det_model = init_detector(det_config, det_checkpoint, device=device.lower())
        # build the pose model from a config file and a checkpoint file
        pose_model = init_pose_model(pose_config, pose_checkpoint, device=device.lower())

        self.models = [
            [det_model, pose_model] for nv in range(nViews)
        ]

    def get_body(self, pose, W, H):
        if pose is None:
            bodies = np.zeros((self.NUM_BODY, 3))
            return bodies, [0, 0, 100, 100, 0]
        poses = pose[self.openpose25_in_23]
        poses[8, :2] = poses[[11, 12], :2].mean(axis=0)
        poses[8, 2] = poses[[11, 12], 2].min(axis=0)
        poses[1, :2] = poses[[5, 6], :2].mean(axis=0)
        poses[1, 2] = poses[[5, 6], 2].min(axis=0)
        return poses, bbox_from_keypoints(poses)

    def get_hand(self, pose, W, H):
        if pose is None:
            bodies = np.zeros((self.NUM_HAND, 3))
            return bodies, [0, 0, 100, 100, 0.]
        return pose, bbox_from_keypoints(pose)

    def get_face(self, pose, W, H):
        if pose is None:
            bodies = np.zeros((self.NUM_FACE, 3))
            return bodies, [0, 0, 100, 100, 0]
        return pose, bbox_from_keypoints(pose)

    def process_body(self, data, results, image_width, image_height):
        keypoints, bbox = self.get_body(results.pose_landmarks, image_width, image_height)
        data['keypoints'] = keypoints
        data['bbox'] = bbox

    def process_hand(self, data, results, image_width, image_height):
        lm = {'Left': None, 'Right': None}
        lm = {'Left': results.left_hand_landmarks, 'Right': results.right_hand_landmarks}
        handl, bbox_handl = self.get_hand(lm['Left'], image_width, image_height)
        handr, bbox_handr = self.get_hand(lm['Right'], image_width, image_height)
        data['handl2d'] = handl.tolist()
        data['bbox_handl2d'] = bbox_handl
        data['handr2d'] = handr.tolist()
        data['bbox_handr2d'] = bbox_handr

    def process_face(self, data, results, image_width, image_height, image=None):
        face2d, bbox_face2d = self.get_face(results.face_landmarks, image_width, image_height)
        data['face2d'] = face2d
        data['bbox_face2d'] = bbox_face2d

    def __call__(self, images):
        annots_all = []
        for nv, image_ in enumerate(images):
            image_height, image_width, _ = image_.shape
            image = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            with Timer('- detect', True):
                det_model = self.models[nv][0]
                pose_model = self.models[nv][1]

                # test a single image, the resulting box is (x1, y1, x2, y2)
                mmdet_results = inference_detector(det_model, image)

                # keep the person class bounding boxes.
                person_results = process_mmdet_results(mmdet_results, 1)

                pose_results, returned_outputs = inference_top_down_pose_model(
                    pose_model,
                    image,
                    person_results,
                    bbox_thr=0.3,
                    format='xyxy',
                    return_heatmap=False,
                    outputs=None)

            data = {
                'personID': 0,
            }

            if len(pose_results[0]) > 0:
                pose_results = pose_results[0]['keypoints']
                results = type('', (object,), {
                    'left_hand_landmarks': pose_results[91:112],
                    'right_hand_landmarks': pose_results[112:133],
                    'face_landmarks': pose_results[23:91],
                    'pose_landmarks': pose_results[0:23]  # body + foot
                })()
            else:
                results = type('', (object,), {
                    'left_hand_landmarks': None,
                    'right_hand_landmarks': None,
                    'face_landmarks': None,
                    'pose_landmarks': None
                })()

            self.process_body(data, results, image_width, image_height)
            self.process_hand(data, results, image_width, image_height)
            with Timer('- face', True):
                self.process_face(data, results, image_width, image_height, image=image)
            annots = {
                'filename': '{}/run.jpg'.format(nv),
                'height': image_height,
                'width': image_width,
                'annots': [
                    data
                ],
                'isKeyframe': False
            }
            # if self.show:
            #     self.vis(image_, annots, nv)
            annots_all.append(annots)
            # results.face_landmarks
        return annots_all


def extract_2d(image_root, annot_root, config):
    from easymocap.estimator.wrapper_base import check_result, save_annot
    force = config.pop('force')
    if check_result(image_root, annot_root) and not force:
        return 0
    from glob import glob
    from os.path import join
    ext = config.pop('ext')
    import os
    from tqdm import tqdm

    detector = Detector(nViews=1, show=False, **config)
    imgnames = sorted(glob(join(image_root, '*' + ext)))
    images = []

    for imgname in imgnames:
        image = cv2.imread(imgname)
        images.append(image)

    for imgname in tqdm(imgnames, desc='{:10s}'.format(os.path.basename(annot_root))):
        base = os.path.basename(imgname).replace(ext, '')
        annotname = join(annot_root, base + '.json')
        image = cv2.imread(imgname)
        annots = detector([image])[0]
        annots['filename'] = os.sep.join(imgname.split(os.sep)[-2:])
        save_annot(annotname, annots)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    path = args.path
    out = args.out
    config = {
        'mmpose': {
            'det_config': 'estimator/MMPose/faster_rcnn_r50_fpn_coco.py',
            'det_checkpoint': 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
            'pose_config': 'estimator/MMPose/hrnet_w48_coco_wholebody_384x288_dark_plus.py',
            'pose_checkpoint': 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth',
            'device': 'cpu',
            'force': False,
            'ext': '.jpg'
        }
    }
    extract_2d(path, out, config['mmpose'])
