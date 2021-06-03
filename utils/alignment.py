"""------------------------------------

* File Name : alignment.py

* Purpose :

* Creation Date : 01-06-2021

* Last Modified : 06:00:54 PM, 03-06-2021

* Created By : Trinh Quang Truong

------------------------------------"""

from skimage import transform as trans
import numpy as np
import sys
import cv2

REFERENCE_FACIAL_POINTS = [[30.29459953, 51.69630051],
                           [65.53179932, 51.50139999],
                           [48.02519989,
                            71.73660278], [33.54930115, 92.3655014],
                           [62.72990036, 92.20410156]]

DEFAULT_CROP_SIZE = (96, 112)

# class FaceWarpException(Exception):
#     def __str__(self):
#         return 'In File {}:{}'.format(
#             __file__, super.__str__(self))


def get_reference_facial_points(output_size=None,
                                inner_padding_factor=0.0,
                                outer_padding=(0, 0),
                                default_square=False):
    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)

    # 0) make the inner region a square
    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    # print('---> default:')
    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)

    if (output_size and output_size[0] == tmp_crop_size[0]
            and output_size[1] == tmp_crop_size[1]):
        #  print(
            #  'output_size == DEFAULT_CROP_SIZE {}: return default reference points'
        #      .format(tmp_crop_size))
        return tmp_5pts

    if (inner_padding_factor == 0 and outer_padding == (0, 0)):
        if output_size is None:
            #  print('No paddings to do: return default reference points')
            return tmp_5pts
        else:
            print('No paddings to do, output_size must be None or {}'.format(
                tmp_crop_size))
            sys.exit(1)

    # check output size
    if not (0 <= inner_padding_factor <= 1.0):
        print('Not (0 <= inner_padding_factor <= 1.0)')
        sys.exit(1)

    if ((inner_padding_factor > 0 or outer_padding[0] > 0
         or outer_padding[1] > 0) and output_size is None):
        output_size = tmp_crop_size * \
            (1 + inner_padding_factor * 2).astype(np.int32)
        output_size += np.array(outer_padding)
        #  print('              deduced from paddings, output_size = ',
         #       output_size)

    if not (outer_padding[0] < output_size[0]
            and outer_padding[1] < output_size[1]):
        print(
            'Not (outer_padding[0] < output_size[0] and outer_padding[1] < output_size[1])'
        )
        sys.exit(1)

    # 1) pad the inner region according inner_padding_factor
    # print('---> STEP1: pad the inner region according inner_padding_factor')
    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2
        tmp_crop_size += np.round(size_diff).astype(np.int32)

    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)

    # 2) resize the padded inner region
    # print('---> STEP2: resize the padded inner region')
    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2
    # print('              crop_size = ', tmp_crop_size)
    # print('              size_bf_outer_pad = ', size_bf_outer_pad)

    if size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[
            1] * tmp_crop_size[0]:
        print(
            'Must have (output_size - outer_padding) = some_scale * (crop_size * (1.0 + inner_padding_factor)'
        )
        sys.exit(1)

    scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
    # print('              resize scale_factor = ', scale_factor)
    tmp_5pts = tmp_5pts * scale_factor
    #    size_diff = tmp_crop_size * (scale_factor - min(scale_factor))
    #    tmp_5pts = tmp_5pts + size_diff / 2
    tmp_crop_size = size_bf_outer_pad
    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)

    # 3) add outer_padding to make output_size
    reference_5point = tmp_5pts + np.array(outer_padding)
    tmp_crop_size = output_size
    # print('---> STEP3: add outer_padding to make output_size')
    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)
    #
    # print('===> end get_reference_facial_points\n')

    return reference_5point


def get_affine_transform_matrix(src_pts, dst_pts):
    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)

    if rank == 3:
        tfm = np.float32([[A[0, 0], A[1, 0], A[2, 0]],
                          [A[0, 1], A[1, 1], A[2, 1]]])
    elif rank == 2:
        tfm = np.float32([[A[0, 0], A[1, 0], 0], [A[0, 1], A[1, 1], 0]])

    return tfm


def warp_and_crop_face(src_img,
                       facial_pts,
                       reference_pts=None,
                       crop_size=(96, 112),
                       align_type='smilarity'):
    if reference_pts is None:
        if crop_size[0] == 96 and crop_size[1] == 112:
            reference_pts = REFERENCE_FACIAL_POINTS
        else:
            default_square = True
            inner_padding_factor = 0.0001
            outer_padding = (0, 0)
            output_size = crop_size

            reference_pts = get_reference_facial_points(
                output_size, inner_padding_factor, outer_padding,
                default_square)

    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape
    if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
        print('reference_pts.shape must be (K,2) or (2,K) and K>2')
        sys.exit(1)

    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
        print('facial_pts.shape must be (K,2) or (2,K) and K>2')
        sys.exit(1)
    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

    if src_pts.shape != ref_pts.shape:
        print('facial_pts and reference_pts must have the same shape')
        sys.exit(1)

    if align_type == 'cv2_affine':
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
    #        print('cv2.getAffineTransform() returns tfm=\n' + str(tfm))
    elif align_type == 'affine':
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
    #        print('get_affine_transform_matrix() returns tfm=\n' + str(tfm))
    else:
        # tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)
        tform = trans.SimilarityTransform()
        tform.estimate(src_pts, ref_pts)
        tfm = tform.params[0:2, :]

    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))

    return face_img


def alignment_img(img,
                  bboxes,
                  points,
                  crop_size=(112, 112),
                  align_type='similarity'):
    results = []
    for point in points:
        print(points)
        results.append(
            warp_and_crop_face(img, facial_pts=point, crop_size=crop_size,
                               align_type=align_type))

    return results
