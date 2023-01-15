import numpy as np
import cv2
from easymocap.mytools import Timer
import numpy as np

from mmpose.datasets import DatasetInfo

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)


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

    def __init__(self, nViews, show=False, **cfg) -> None:
        self.nViews = nViews
        self.show = show
        self.NUM_BODY = 25
        self.model = cfg['model']
        if self.model == 'coco-wholebody':
            self.openpose25_mapping = [0, 0, 6, 8, 10, 5, 7, 9, 0, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 17, 18, 19, 20, 21,
                                     22]
        elif self.model == 'halpe':
            self.openpose25_mapping = [0, 18, 6, 8, 10, 5, 7, 9, 19, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 20, 22, 24, 21, 23,
                                     25]
        elif self.model == 'coco':
            self.openpose25_mapping = [0, 0, 6, 8, 10, 5, 7, 9, 0, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 15, 15, 15, 16, 16,
                                     16]

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
        poses = pose[self.openpose25_mapping]
        if self.model.startswith('coco'):
            poses[8, :2] = poses[[9, 12], :2].mean(axis=0)
            poses[8, 2] = poses[[9, 12], 2].min(axis=0)
            poses[1, :2] = poses[[2, 5], :2].mean(axis=0)
            poses[1, 2] = poses[[2, 5], 2].min(axis=0)
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

                dataset = pose_model.cfg.data['test']['type']
                dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
                if dataset_info is None:
                    print(
                        'Please set `dataset_info` in the config.'
                        'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                        DeprecationWarning)
                else:
                    dataset_info = DatasetInfo(dataset_info)
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
                    dataset=dataset,
                    dataset_info=dataset_info,
                    return_heatmap=False,
                    outputs=None)

            data = {
                'personID': 0,
            }

            if len(pose_results[0]) > 0:
                pose_results = pose_results[0]['keypoints']
                if self.model == 'coco-wholebody':
                    results = type('', (object,), {
                        'left_hand_landmarks': pose_results[91:112],
                        'right_hand_landmarks': pose_results[112:133],
                        'face_landmarks': pose_results[23:91],
                        'pose_landmarks': pose_results[0:23]  # body + foot
                    })()
                elif self.model == 'halpe':
                    results = type('', (object,), {
                        'left_hand_landmarks': pose_results[94:115],
                        'right_hand_landmarks': pose_results[115:136],
                        'face_landmarks': pose_results[26:94],
                        'pose_landmarks': pose_results[0:26]  # body + foot
                    })()
                elif self.model == 'coco':
                    results = type('', (object,), {
                        'left_hand_landmarks': None,
                        'right_hand_landmarks': None,
                        'face_landmarks': None,
                        'pose_landmarks': pose_results[0:17]  # just body
                    })()
                else:
                    results = type('', (object,), {
                        'left_hand_landmarks': None,
                        'right_hand_landmarks': None,
                        'face_landmarks': None,
                        'pose_landmarks': None
                    })()
            else:
                results = type('', (object,), {
                    'left_hand_landmarks': None,
                    'right_hand_landmarks': None,
                    'face_landmarks': None,
                    'pose_landmarks': None
                })()

            self.process_body(data, results, image_width, image_height)

            # coco has no face or hands even feet
            if self.model != 'coco':
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
            'det_config': 'easymocap/estimator/MMPose/mmdet_configs/configs/yolox/yolox_x_8x8_300e_coco.py',
            'det_checkpoint': 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth',
            'pose_config': 'easymocap/estimator/MMPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py',
            'pose_checkpoint': 'easymocap/estimator/MMPose/models/vitpose-l-multi-coco.pth',
            'device': 'cuda',
            'force': False,
            'ext': '.jpg',
            'model': 'coco'
        }
    }
    extract_2d(path, out, config['mmpose'])
