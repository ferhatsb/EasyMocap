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

    body = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
            [13, 14], [15, 16]]
    foot = [[17, 20], [18, 21], [19, 22]]

    face = [[23, 39], [24, 38], [25, 37], [26, 36], [27, 35], [28, 34],
            [29, 33], [30, 32], [40, 49], [41, 48], [42, 47], [43, 46],
            [44, 45], [54, 58], [55, 57], [59, 68], [60, 67], [61, 66],
            [62, 65], [63, 70], [64, 69], [71, 77], [72, 76], [73, 75],
            [78, 82], [79, 81], [83, 87], [84, 86], [88, 90]]

    hand = [[91, 112], [92, 113], [93, 114], [94, 115], [95, 116],
            [96, 117], [97, 118], [98, 119], [99, 120], [100, 121],
            [101, 122], [102, 123], [103, 124], [104, 125], [105, 126],
            [106, 127], [107, 128], [108, 129], [109, 130], [110, 131],
            [111, 132]]

    def __init__(self, nViews, show=False, **cfg) -> None:
        self.nViews = nViews
        self.show = show
        self.NUM_BODY = 25
        self.openpose25_in_23 = [0, 0, 6, 8, 10, 5, 7, 9, 0, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 20, 21, 22, 17, 18, 19]

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

    @staticmethod
    def to_array(pose, W, H, start=0):
        # N = len(pose) - start
        # res = np.zeros((N, 3))
        # for i in range(start, len(pose)):
        #     res[i - start, 0] = pose[i][0]
        #     res[i - start, 1] = pose[i][1]
        #     res[i - start, 2] = pose[i][2]
        return pose

    def get_body(self, pose, W, H):
        if pose is None:
            bodies = np.zeros((self.NUM_BODY, 3))
            return bodies, [0, 0, 100, 100, 0]
        poses = self.to_array(pose, W, H)
        poses = poses[self.openpose25_in_23]
        poses[8, :2] = poses[[11, 12], :2].mean(axis=0)
        poses[8, 2] = poses[[11, 12], 2].min(axis=0)
        poses[1, :2] = poses[[5, 6], :2].mean(axis=0)
        poses[1, 2] = poses[[5, 6], 2].min(axis=0)
        return poses, bbox_from_keypoints(poses)

    def get_hand(self, pose, W, H):
        if pose is None:
            bodies = np.zeros((self.NUM_HAND, 3))
            return bodies, [0, 0, 100, 100, 0.]
        poses = self.to_array(pose, W, H)
        return poses, bbox_from_keypoints(poses)

    def get_face(self, pose, W, H):
        if pose is None:
            bodies = np.zeros((self.NUM_FACE, 3))
            return bodies, [0, 0, 100, 100, 0]
        poses = self.to_array(pose, W, H)
        return poses, bbox_from_keypoints(poses)

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
                    'left_hand_landmarks': pose_results[112:133],
                    'right_hand_landmarks': pose_results[91:112],
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
