import cv2
import numpy as np
from imageint.utils.preprocess import rgb_to_srgb as _tonemap_srgb



def mse_to_psnr(mse):
    """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
    return -10. / np.log(10.) * np.log(mse)


def calc_PSNR(img_pred, img_gt, mask_gt, max_value=1, tonemapping=True, divide_mask=True):
    '''
        calculate the PSNR between the predicted image and ground truth image.
        a scale is optimized to get best possible PSNR.
        images are clip by max_value_ratio.
        params:
        img_pred: numpy.ndarray of shape [H, W, 3]. predicted HDR image.
        img_gt: numpy.ndarray of shape [H, W, 3]. ground truth HDR image.
        mask_gt: numpy.ndarray of shape [H, W]. ground truth foreground mask.
        max_value: Float. the maximum value of the ground truth image clipped to.
            This is designed to prevent the result being affected by too bright pixels.
        tonemapping: Bool. Whether the images are tone-mapped before comparion.
        divide_mask: Bool. Whether the mse is divided by the foreground area.
    '''
    if mask_gt.ndim == 3:
        mask_gt = mask_gt[..., 0]
    if mask_gt.dtype == np.float32:
        mask_gt = (mask_gt * 255).clip(0, 255).astype(np.uint8)
    # shrink mask by a small margin to prevent inaccurate mask boundary.
    kernel = np.ones((5, 5), np.uint8)
    mask_gt = cv2.erode(mask_gt, kernel)
    mask_gt = (mask_gt > 127).astype(np.float32)
    img_pred *= mask_gt[..., None]
    img_gt *= mask_gt[..., None]

    invalid = False
    for c in range(img_pred.shape[-1]):
        pred_median = np.median(img_pred[..., c][np.where(mask_gt > 0.5)])
        if pred_median <= 1e-6:
            invalid = True
    if invalid:
        print('invalid prediction from calc_PSNR')
        img_pred = np.ones_like(img_pred)

    # scale the prediction to the ground truth channel-wisely to eliminate the ambiguity in inverse rendering
    for c in range(img_pred.shape[-1]):
        gt_median = np.median(img_gt[..., c][np.where(mask_gt > 0.5)])
        pred_median = np.median(img_pred[..., c][np.where(mask_gt > 0.5)])
        img_pred[..., c] *= gt_median / pred_median
        if not tonemapping:  # image in linear space are usually too dark, need to re-normalize
            img_pred[..., c] *= 0.25 / gt_median
            img_gt[..., c] *= 0.25 / gt_median

            # if not tonemapping:
    #     imageio.imsave("./rescaled.exr", img_pred)
    #     imageio.imsave("./rescaled_gt.exr", img_gt)

    # clip the prediction and the gt img by the maximum_value
    img_pred = np.clip(img_pred, 0, max_value)
    img_gt = np.clip(img_gt, 0, max_value)

    if tonemapping:
        img_pred = _tonemap_srgb(img_pred)
        img_gt = _tonemap_srgb(img_gt)
        # imageio.imsave("./rescaled.png", (img_pred*255).clip(0,255).astype(np.uint8))
        # imageio.imsave("./rescaled_gt.png", (img_gt*255).clip(0,255).astype(np.uint8))

    if not divide_mask:
        mse = ((img_pred - img_gt) ** 2).mean()
    else:
        mse = ((img_pred - img_gt) ** 2).sum() / mask_gt.sum()
    return mse_to_psnr(mse)


# pred_img = imageio.imread("./inputs.exr")
# gt_img = imageio.imread("./target.exr")
# gt_mask = imageio.imread("./mask.png")
#
# divide_mask = True
# PSNR_hdr = calc_PSNR(pred_img, gt_img, gt_mask, 4, tonemapping=False, divide_mask=divide_mask)
# PSNR_ldr = calc_PSNR(pred_img, gt_img, gt_mask, 1, tonemapping=True, divide_mask=divide_mask)
#
# print("PSNR_HD = %2.4f, PSNR_LD = %2.4f" % (PSNR_hdr, PSNR_ldr))


def erode_mask(mask, target_size):
    if mask.ndim == 3:
        mask = mask[...,0]
    if mask.dtype == np.float32:
        mask = (mask*255).clip(0, 255).astype(np.uint8)
    # shrink mask by a small margin to prevent inaccurate mask boundary.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    if target_size is not None:
        mask = cv2.resize(mask, (target_size, target_size))
    return (mask > 127).astype(np.float32)


def calc_normal_distance(normal_pred, normal_gt, mask_gt):
    '''
        params:
        normal_pred: numpy.ndarray of shape [H, W, 3].
        normal_gt: numpy.ndarray of shape [H, W, 3].
        mask_gt: numpy.ndarray of shape [H, W]. ground truth foreground mask.
    '''
    assert normal_pred.shape == normal_gt.shape, (normal_pred.shape, normal_gt.shape)
    mask_gt = erode_mask(mask_gt, normal_pred.shape[0])
    cos_dist = (1-(normal_pred * normal_gt).sum(axis=-1)) * mask_gt
    return float(cos_dist.sum() / mask_gt.sum())


# def calc_depth_distance(depth_pred, depth_gt, mask_gt):
#     '''
#         params:
#         depth_pred: numpy.ndarray of shape [H, W].
#         depth_gt: numpy.ndarray of shape [H, W].
#         mask_gt: numpy.ndarray of shape [H, W]. ground truth foreground mask.
#     '''
#     assert depth_pred.shape == depth_gt.shape
#     mask_gt = erode_mask(mask_gt, depth_pred.shape[0])
#     gt_median = np.median(depth_gt[np.where(mask_gt>0.5)])
#     pred_median = np.median(depth_pred[np.where(mask_gt>0.5)])
#     depth_pred *= gt_median / (pred_median + 1e-9)
#     return float((((depth_pred - depth_gt)**2) * mask_gt).sum() / mask_gt.sum())


def calc_depth_distance(depth_pred, depth_gt, mask_gt):
    '''
        params:
        depth_pred: numpy.ndarray of shape [H, W].
        depth_gt: numpy.ndarray of shape [H, W].
        mask_gt: numpy.ndarray of shape [H, W]. ground truth foreground mask.
    '''
    assert depth_pred.shape == depth_gt.shape
    mask_gt = erode_mask(mask_gt, depth_pred.shape[0])
    depth_gt_masked = depth_gt[np.where(mask_gt>0.5)]
    depth_pred_masked = depth_pred[np.where(mask_gt>0.5)]
    if (depth_pred_masked ** 2).sum() <= 1e-6:
        depth_pred = np.ones_like(depth_gt) # np.maximum(depth_pred, .99 * np.min(depth_gt_masked) * np.ones_like(depth_pred))
        depth_pred_masked = depth_pred[np.where(mask_gt>0.5)]
    scale = (depth_gt_masked * depth_pred_masked).sum() / (depth_pred_masked**2).sum()
    depth_pred = scale * depth_pred
    # depth_pred *= gt_median / (pred_median+1e-9)
    return float((((depth_pred - depth_gt)**2) * mask_gt).sum() / mask_gt.sum())


if __name__ == "__main__":
    import pyexr
    from pathlib import Path
    from imageint.utils.preprocess import load_rgb_png, load_rgb_exr, srgb_to_rgb, load_hdr_rgba, rgb_to_srgb, load_mask_png, cv2_downsize
    item = {
        'output_depth': '/viscam/projects/imageint/yzzhang/imageint/imageint/third_party/neuralpil/evals/test/scene002_obj008_grogu/0530/test_imgs_384000/0_fine_depth_processed.exr',
        'target_depth': '/viscam/projects/imageint/capture_scene_data/data/scene002_obj008_grogu/final_output/geometry_outputs/z_maps/0060.npy',
    }
    input_depth = pyexr.open(item['output_depth']).get().squeeze()
    target_depth = cv2_downsize(np.load(item['target_depth']), downsize_factor=4)
    mask = load_mask_png(Path(item['target_depth']).parent.parent.parent / 'blender_format_LDR/test_mask' / Path(
        item['target_depth']).with_suffix('.png').name, downsize_factor=4).astype(np.float32)

    import ipdb; ipdb.set_trace()
    calc_depth_distance(np.zeros_like(input_depth), target_depth, mask)
    calc_depth_distance(input_depth, target_depth, mask)
