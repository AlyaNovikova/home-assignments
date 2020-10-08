#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli, filter_frame_corners


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


class Tracker:
    MAX_CORNERS = 0
    QUALITY_LEVEL = 0.01
    MIN_DISTANCE = 10
    BLOCK_SIZE = 3
    CORNER_SIZES = 10
    LEVELS_IN_PYRAMID = 5
    WIN_SIZE = (15, 15)

    def corners_on_one_level(self, img, level):
        corners = cv2.goodFeaturesToTrack(img,
                                          maxCorners=self.MAX_CORNERS,
                                          qualityLevel=self.QUALITY_LEVEL,
                                          minDistance=self.MIN_DISTANCE * 2 ** level,
                                          blockSize=self.BLOCK_SIZE,
                                          useHarrisDetector=False
                                          )
        if corners is None:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        corners = np.squeeze(corners, 1)
        return corners, np.full(corners.shape[0], self.CORNER_SIZES)

    def scale_corners(self, corners, sizes, level):
        return corners * 2 ** level, sizes * 2 ** level

    def addCorners(self, img_h, img_w, corners1, sizes1, ids1, corners2, sizes2, ids2):
        mask = np.zeros((img_h, img_w))

        for corner, size in zip(corners1, sizes1):
            mask = cv2.circle(mask, (int(corner[0]), int(corner[1])), int(size), color=1, thickness=-1)

        inds = []
        for i, corner in enumerate(corners2):
            coord = (int(corner[1]), int(corner[0]))
            if not (0 <= coord[0] < img_h and 0 <= coord[1] < img_w):
                continue
            if mask[coord] == 0:
                inds.append(i)

        corners2 = corners2[inds]
        sizes2 = sizes2[inds]
        ids2 = ids2[inds]

        return np.concatenate([corners1, corners2], 0), \
               np.concatenate([sizes1, sizes2], 0), \
               np.concatenate([ids1, ids2], 0)

    def filterCorners(self, corners, img_h, img_w):
        mask = np.zeros((img_h, img_w))

        inds = []
        for i, (_, corner, size) in enumerate(zip(corners.ids, corners.points, corners.sizes)):
            coord = (int(corner[1]), int(corner[0]))
            if not (0 <= coord[0] < img_h and 0 <= coord[1] < img_w):
                continue
            if mask[coord] == 0:
                mask = cv2.circle(mask, coord[::-1], int(size * 0.5), color=1, thickness=-1)
                inds.append(i)

        return filter_frame_corners(corners, np.array(inds))

    def find_corners(self, img, last_id):
        corners = np.zeros((0, 2))
        sizes = np.zeros(0, dtype=np.int0)
        ids = np.zeros(0, dtype=np.int0)

        for level in range(self.LEVELS_IN_PYRAMID):
            new_corners, new_sizes = self.corners_on_one_level(img, level)
            new_corners, new_sizes = self.scale_corners(new_corners, new_sizes, level)
            new_ids = np.arange(len(new_corners)) + last_id + 1

            last_id += len(new_corners)

            corners, sizes, ids = self.addCorners(img.shape[0], img.shape[1],
                                                  corners, sizes, ids,
                                                  new_corners, new_sizes, new_ids)

            img = cv2.pyrDown(img)

        return corners, sizes, ids

    def tracking(self, img1, img2, corners, sizes, ids):
        corners = np.float32(corners)
        new_corners = np.zeros_like(corners)

        img1 = np.uint8(img1 * 255)
        img2 = np.uint8(img2 * 255)

        new_corners, status, err = cv2.calcOpticalFlowPyrLK(img1,
                                                            img2,
                                                            corners,
                                                            new_corners,
                                                            winSize=self.WIN_SIZE,
                                                            maxLevel=self.LEVELS_IN_PYRAMID
                                                            )

        filter = np.squeeze(status.astype(np.bool), 1)
        new_corners = new_corners[filter]
        new_sizes = sizes[filter]
        new_ids = ids[filter]
        return new_corners, new_sizes, new_ids


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    tracker = Tracker()
    img_prev = None
    last_id = 0

    for frame, img_curr in enumerate(frame_sequence):
        if img_prev is None:
            coords, sizes, ids = tracker.find_corners(img_curr, 0)
            last_id += len(coords)
        if img_prev is not None:
            coords, sizes, ids = tracker.tracking(img_prev, img_curr,
                                                  corners.points,
                                                  corners.sizes[:, 0],
                                                  corners.ids[:, 0])
            new_coords, new_sizes, new_ids = tracker.find_corners(img_curr, last_id)
            last_id += len(new_coords)

            coords, sizes, ids = tracker.addCorners(img_curr.shape[0], img_curr.shape[1],
                                                    coords, sizes, ids,
                                                    new_coords, new_sizes, new_ids)

        corners = FrameCorners(ids, coords, sizes)
        corners = tracker.filterCorners(corners, img_curr.shape[0], img_curr.shape[1])

        builder.set_corners_at_frame(frame, corners)
        img_prev = img_curr


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
