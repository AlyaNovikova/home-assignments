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
from _corners import dump, load, draw, without_short_tracks, create_cli


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
    MAX_CORNERS = 1000
    QUALITY_LEVEL = 0.03
    MIN_DISTANCE = 10
    BLOCK_SIZE = 3
    CORNER_SIZES = 10
    LEVELS_IN_PYRAMID = 3
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

    # Add a new corner only if there are no old corners inside
    def addCorners(self, corners1, size1, corners2, size2):
        if len(corners2) == 0:
            return corners1, size1
        if len(corners1) == 0:
            return corners2, size2

        diffs = corners2[:, None, :] - corners1[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        min_dists = np.min(dists, 1)
        filter = min_dists < size2

        corners2, size2 = corners2[~filter], size2[~filter]

        return np.concatenate([corners1, corners2], 0), np.concatenate([size1, size2], 0)

    def find_corners(self, img):
        corners = np.zeros((0, 2))
        sizes = np.zeros(0, dtype=np.int0)

        for level in range(self.LEVELS_IN_PYRAMID):
            new_corners, new_sizes = self.corners_on_one_level(img, level)
            new_corners, new_sizes = self.scale_corners(new_corners, new_sizes, level)

            corners, sizes = self.addCorners(corners, sizes, new_corners, new_sizes)

            img = cv2.pyrDown(img)

        return corners, sizes

    def tracking(self, img1, img2, corners, sizes):
        corners = np.float32(corners)
        new_corners = np.zeros_like(corners)

        img1 = np.uint8(img1 * 255)
        img2 = np.uint8(img2 * 255)

        new_corners, status, err = cv2.calcOpticalFlowPyrLK(img1,
                                                            img2,
                                                            corners,
                                                            new_corners,
                                                            winSize=self.WIN_SIZE,
                                                            maxLevel=self.LEVELS_IN_PYRAMID,
                                                            criteria=(
                                                                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,
                                                                0.03),
                                                            minEigThreshold=0.001
                                                            )

        filter = np.squeeze(status.astype(np.bool), 1)
        new_corners = new_corners[filter]
        new_sizes = sizes[filter]
        return new_corners, new_sizes


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    tracker = Tracker()
    img_prev = None

    for frame, img_curr in enumerate(frame_sequence):
        if img_prev is None:
            coords, sizes = tracker.find_corners(img_curr)
        if img_prev is not None:
            coords, sizes = tracker.tracking(img_prev, img_curr, coords, sizes)
            new_coords, new_sizes = tracker.find_corners(img_curr)
            coords, sizes = tracker.addCorners(coords, sizes, new_coords, new_sizes)

        corners = FrameCorners(np.arange(len(coords)), coords, sizes)

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
