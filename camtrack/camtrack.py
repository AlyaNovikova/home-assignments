#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    Correspondences,
    TriangulationParameters,
    build_correspondences,
    triangulate_correspondences,
    compute_reprojection_errors,
    rodrigues_and_translation_to_view_mat3x4
)

class Tracker:
    TRIANGULATION_INIT = TriangulationParameters(max_reprojection_error=2.,
                                                 min_triangulation_angle_deg=0.01,
                                                 min_depth=0.001)
    TRIANGULATION = TriangulationParameters(max_reprojection_error=1.,
                                            min_triangulation_angle_deg=2.,
                                            min_depth=0.1)

    CONFIDENCE = 0.999
    FRAMES_FOR_RETRIANGULATION = 10
    RETRIANGULATION_FREQUENCY = 29
    RETRIANGULATION_REPETITIONS = 10
    MAX_RETRIANGULATION_ERROR = 0.4
    MAX_FRAMES = 20

    def init(self, corner_storage, known_view_1, known_view_2, intrinsic_mat):
        frame_1, frame_2 = known_view_1[0], known_view_2[0]
        view_mat_1, view_mat_2 = pose_to_view_mat3x4(known_view_1[1]), pose_to_view_mat3x4(known_view_2[1])

        coors = build_correspondences(corner_storage[frame_1], corner_storage[frame_2])
        points3d, corrs_ids, median_cos = triangulate_correspondences(coors,
                                                                      view_mat_1, view_mat_2,
                                                                      intrinsic_mat,
                                                                      self.TRIANGULATION_INIT)
        frame_count = len(corner_storage)
        view_mats = [None] * frame_count
        view_mats[frame_1], view_mats[frame_2] = view_mat_1, view_mat_2

        print(f'Initialization. Cloud contains {len(points3d)} points')

        return view_mats, PointCloudBuilder(corrs_ids, points3d)

    def solve_pnp(self, corners, point_cloud_builder, intrinsic_mat):
        ids2d = corners.ids.flatten()
        ids3d = point_cloud_builder.ids.flatten()
        _, indices3d, indices2d = np.intersect1d(ids3d, ids2d, return_indices=True)
        points2d = corners.points[indices2d]
        points3d = point_cloud_builder.points[indices3d]

        if len(points2d) < 4:
            return False, None, None, None

        retval, rvec, tvec, inliers = cv2.solvePnPRansac(points3d, points2d,
                                                         intrinsic_mat,
                                                         distCoeffs=None,
                                                         confidence=self.CONFIDENCE,
                                                         flags=cv2.SOLVEPNP_EPNP)
        if not retval:
            return False, None, None, None

        ids = np.array(inliers).flatten()
        _, rvec, tvec = cv2.solvePnP(points3d[ids], points2d[ids],
                                     intrinsic_mat,
                                     distCoeffs=None,
                                     rvec=rvec,tvec=tvec,
                                     useExtrinsicGuess=True,
                                     flags=cv2.SOLVEPNP_ITERATIVE)

        return retval, rvec, tvec, inliers

    def update_corners(self, corners, ids):
        res = []
        for parameter in corners:
            res.append(parameter[ids])
        return FrameCorners(*res)

    def update(self, i, corners, corner_storage, point_cloud_builder, view_mats, intrinsic_mat):
        for j in range(len(corner_storage)):
            if i == j or view_mats[j] is None:
                continue
            coors = build_correspondences(corner_storage[j], corners)
            if len(coors) == 0:
                continue

            points, ids, median_cos = triangulate_correspondences(coors,
                                                                  view_mats[j], view_mats[i],
                                                                  intrinsic_mat,
                                                                  self.TRIANGULATION)

            if len(points) != 0:
                print(f'\t Triangulated {len(points)} points in {j} frame')

            point_cloud_builder.my_add_points(ids, points)

    def retriangulate(self, corner_storage, view_mats, point_cloud_builder, intrinsic_mat):
        frame_count = len(corner_storage)
        new_points, new_ids = [], []
        for point_id in point_cloud_builder.ids:
            id = point_id[0]
            frames, corners, mats = [], [], []
            for frame in range(frame_count):
                ids = corner_storage[frame].ids
                points = corner_storage[frame].points
                if view_mats[frame] is not None and id in ids:
                    frames.append(frame)
                    mats.append(view_mats[frame])
                    corners.append(points[np.where(ids.flatten() == id)[0][0]])

            if len(frames) < self.FRAMES_FOR_RETRIANGULATION:
                continue

            max_inliers = 0
            res_point = None

            frames, mats, corners = np.array(frames), np.array(mats), np.array(corners)
            max_frames = min(self.MAX_FRAMES, len(frames))
            inds = np.random.choice(len(frames), max_frames, replace=False)
            frames, mats, corners = frames[inds], mats[inds], corners[inds]

            for _ in range(self.RETRIANGULATION_REPETITIONS):
                i = np.random.choice(len(frames))
                j = np.random.choice(len(frames))

                coors = Correspondences(np.array([id]), np.array([corners[i]]), np.array([corners[j]]))

                point, ids, median_cos = triangulate_correspondences(coors,
                                                                     mats[i], mats[j],
                                                                     intrinsic_mat,
                                                                     self.TRIANGULATION)

                if len(point) == 0:
                    continue
                errors = np.array([compute_reprojection_errors(point,
                                                               corners[i_frame],
                                                               intrinsic_mat @ view_mats[frame]).flatten()[0]
                                   for i_frame, frame in enumerate(frames)])

                inliers = errors[errors < self.MAX_RETRIANGULATION_ERROR]

                if max_inliers < len(inliers):
                    max_inliers = len(inliers)
                    res_point = point

            if res_point is not None:
                new_points.append(res_point)
                new_ids.append(id)

        if len(new_points) != 0:
            print(f'Retriangulated {len(new_points)} points')
            point_cloud_builder.my_update_points(np.array(new_ids), np.array(new_points))

    def track(self, corner_storage, view_mats, point_cloud_builder, intrinsic_mat):
        frame_count = len(corner_storage)
        cnt = 0
        while True:
            was_updated = False
            for i in range(frame_count):
                if view_mats[i] is not None:
                    continue

                cur_corners = corner_storage[i]
                retval, rvec, tvec, inliers = self.solve_pnp(cur_corners, point_cloud_builder, intrinsic_mat)

                if not retval or inliers is None:
                    continue
                ids = inliers.flatten()

                if len(ids) == 0:
                    continue

                print(f'Now on frame {i}')
                print(f'Current progress: {np.sum([i is not None for i in view_mats])}  frames out of {len(view_mats)}')
                print(f'\t Found {len(ids)} inliers')

                was_updated = True
                view_mats[i] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

                self.update(i,
                            self.update_corners(cur_corners, ids), corner_storage,
                            point_cloud_builder,
                            view_mats,
                            intrinsic_mat)
                print(f'\t Cloud contains {len(point_cloud_builder.points)} points')

                if cnt % self.RETRIANGULATION_FREQUENCY == 0:
                    self.retriangulate(corner_storage, view_mats, point_cloud_builder, intrinsic_mat)
                cnt += 1

            if not was_updated:
                break


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    np.random.seed(1)

    tracker = Tracker()
    view_mats, point_cloud_builder = tracker.init(corner_storage, known_view_1, known_view_2, intrinsic_mat)
    tracker.track(corner_storage, view_mats, point_cloud_builder, intrinsic_mat)
    print(f'Cloud contains {len(point_cloud_builder.points)} points')

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
