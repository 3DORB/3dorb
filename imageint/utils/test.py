import datetime
from PIL import Image
import os
import json
import glob
import numpy as np
from imageint.utils.ppp import build_from_config, list_of_dicts__to__dict_of_lists, overwrite_cfg
from imageint.utils.constant import load_metadata
from typing import Dict, List
from imageint.constant import PROJ_ROOT
from pathlib import Path
from imageint.utils.metrics import calc_PSNR as _psnr, calc_depth_distance, calc_normal_distance, erode_mask
from imageint.utils.preprocess import load_rgb_png, load_rgb_exr, srgb_to_rgb, load_hdr_rgba, rgb_to_srgb, load_mask_png, cv2_downsize
import logging

try:
    import torch
    import pyexr
    from lpips import LPIPS
    from kornia.losses import ssim_loss
    _lpips = None
except Exception as _:
    LPIPS = None
    ssim_loss = None


logger = logging.getLogger(__name__)


METADATA = load_metadata()
LEADERBOARD_DIR = os.path.join(PROJ_ROOT, 'logs/leaderboard')
DEBUG = os.getenv('DEBUG') == '1'
OVERWRITE = os.getenv('OVERWRITE') == '1'
OVERWRITE_VIEW = os.getenv('OVERWRITE_VIEW') == '1'
OVERWRITE_LIGHT = os.getenv('OVERWRITE_LIGHT') == '1'
OVERWRITE_GEOMETRY = os.getenv('OVERWRITE_GEOMETRY') == '1'
OVERWRITE_MATERIAL = os.getenv('OVERWRITE_MATERIAL') == '1'
NO_SCORE_VIEW = os.getenv('NO_SCORE_VIEW') == '1'
NO_SCORE_LIGHT = os.getenv('NO_SCORE_LIGHT') == '1'
NO_SCORE_GEOMETRY = os.getenv('NO_SCORE_GEOMETRY') == '1'
NO_SCORE_MATERIAL = os.getenv('NO_SCORE_MATERIAL') == '1'
logger.info(f'DEBUG={DEBUG}, OVERWRITE={OVERWRITE}, OVERWRITE_VIEW={OVERWRITE_VIEW}, OVERWRITE_LIGHT={OVERWRITE_LIGHT}, OVERWRITE_GEOMETRY={OVERWRITE_GEOMETRY}, NO_SCORE_VIEW={NO_SCORE_VIEW}, NO_SCORE_LIGHT={NO_SCORE_LIGHT}, NO_SCORE_GEOMETRY={NO_SCORE_GEOMETRY}, NO_SCORE_MATERIAL={NO_SCORE_MATERIAL}')


def assert_inputs_target(inputs: np.ndarray, target: np.ndarray, mask: np.ndarray):
    # inputs and targets have range [0, 1]
    assert inputs.dtype == np.float32, inputs.dtype
    assert target.dtype == np.float32, target.dtype
    assert mask.dtype == np.float32, mask.dtype
    assert inputs.shape == target.shape == (512, 512, 3), (inputs.shape, target.shape)
    assert mask.shape == (512, 512), mask.shape


def lpips(inputs: np.ndarray, target: np.ndarray, mask: np.ndarray):
    if LPIPS is None:
        return np.nan
    global _lpips
    if _lpips is None:
        _lpips = LPIPS(net='vgg').cuda()
    inputs = rgb_to_srgb(inputs)
    target = rgb_to_srgb(target)

    mask = erode_mask(mask, None)
    inputs = inputs * mask[:, :, None]
    target = target * mask[:, :, None]

    inputs = torch.tensor(inputs, dtype=torch.float32, device='cuda').permute(2, 0, 1).unsqueeze(0)
    target = torch.tensor(target, dtype=torch.float32, device='cuda').permute(2, 0, 1).unsqueeze(0)
    return _lpips(inputs, target, normalize=True).item()


def ssim(inputs: np.ndarray, target: np.ndarray, mask: np.ndarray):
    if ssim_loss is None:
        return np.nan

    mask = erode_mask(mask, None)
    inputs = inputs * mask[:, :, None]
    target = target * mask[:, :, None]

    # image_pred and image_gt: (1, 3, H, W) in range [0, 1]
    inputs = torch.tensor(inputs, dtype=torch.float32, device='cuda').permute(2, 0, 1).unsqueeze(0)
    target = torch.tensor(target, dtype=torch.float32, device='cuda').permute(2, 0, 1).unsqueeze(0)
    dssim_ = ssim_loss(inputs, target, 3).item()  # dissimilarity in [0, 1]
    return 1 - 2 * dssim_  # in [-1, 1]


def psnr(inputs: np.ndarray, target: np.ndarray, mask: np.ndarray):
    mask = (mask > .5).astype(np.float32)
    return _psnr(inputs, target, mask, 4, tonemapping=False, divide_mask=False), _psnr(inputs, target, mask, 1, tonemapping=True, divide_mask=False)


def compute_similarity(input_hdr: np.ndarray, input_ldr: np.ndarray, target_hdr: np.ndarray, mask: np.ndarray):
    target_ldr = rgb_to_srgb(target_hdr)
    assert_inputs_target(input_hdr, target_hdr, mask)
    assert_inputs_target(input_ldr, target_ldr, mask)
    return {
        'psnr_hdr': psnr(input_hdr, target_hdr, mask)[0],
        'lpips': lpips(input_ldr, target_ldr, mask),
        'ssim': ssim(input_ldr, target_ldr, mask),
        'psnr_ldr': psnr(input_ldr, target_ldr, mask)[1],
    }


def compute_similarity_ldr(input_ldr: np.ndarray, target_ldr: np.ndarray, mask: np.ndarray):
    assert_inputs_target(input_ldr, target_ldr, mask)
    return {
        'lpips': lpips(input_ldr, target_ldr, mask),
        'ssim': ssim(input_ldr, target_ldr, mask),
        'psnr_ldr': psnr(input_ldr, target_ldr, mask)[1],
    }


def psnr_not_used(inputs: np.ndarray, target: np.ndarray, mask: np.ndarray):
    Image.fromarray((rgb_to_srgb(inputs) * 255).astype(np.uint8)).save('/viscam/projects/imageint/yzzhang/tmp/inputs.png')
    Image.fromarray((rgb_to_srgb(target) * 255).astype(np.uint8)).save('/viscam/projects/imageint/yzzhang/tmp/targets.png')
    Image.fromarray((mask * 255).astype(np.uint8)).save('/viscam/projects/imageint/yzzhang/tmp/mask.png')
    Image.fromarray((rgb_to_srgb(inputs * mask[:, :, None]) * 255).astype(np.uint8)).save('/viscam/projects/imageint/yzzhang/tmp/inputs_masked.png')
    Image.fromarray((rgb_to_srgb(target * mask[:, :, None]) * 255).astype(np.uint8)).save('/viscam/projects/imageint/yzzhang/tmp/targets_masked.png')
    Image.fromarray((rgb_to_srgb(np.abs(inputs - target) * mask[:, :, None]) * 255).astype(np.uint8)).save('/viscam/projects/imageint/yzzhang/tmp/diff_masked.png')

    assert inputs.shape == target.shape == (512, 512, 3), (inputs.shape, target.shape)
    inputs = rgb_to_srgb(inputs).clip(0, 1)
    target = rgb_to_srgb(target).clip(0, 1)

    mse = ((inputs - target) ** 2).mean()
    ret = -10. / np.log(10.) * np.log(mse)
    return ret


def compute_metrics_material(results: List) -> Dict:
    ret = []
    for item in results:
        target_rgb_ldr = load_rgb_png(item['target_image'], downsize_factor=4)
        target_alpha = load_mask_png(os.path.join(os.path.dirname(item['target_image']), '../../blender_format_LDR/test_mask', os.path.basename(item['target_image'])), downsize_factor=4).astype(np.float32)
        if item['output_image'] is None:
            input_rgb_ldr = np.ones((512, 512, 3), dtype=np.float32)
        elif item['output_image'].endswith('.exr'):
            input_rgb_hdr = load_rgb_exr(item['output_image'])
            input_rgb_ldr = rgb_to_srgb(input_rgb_hdr)
        elif item['output_image'].endswith('.png'):
            input_rgb_ldr = load_rgb_png(item['output_image'])
        else:
            raise NotImplementedError(item['output_image'])
        ret.append(compute_similarity_ldr(input_rgb_ldr, target_rgb_ldr, target_alpha))

    ret = list_of_dicts__to__dict_of_lists(ret)
    if LPIPS is not None:
        for k, v in ret.items():
            if np.isnan(v).any():
                logger.error(f'NAN in {k}, {v}, {np.asarray(results)[np.isnan(v)]}')
                import ipdb; ipdb.set_trace()
                # ret[k] = np.asarray(v)[~np.isnan(v)]
    ret = {k: np.mean(v) for k, v in ret.items()}
    return ret


def compute_metrics_image_similarity(results: List) -> Dict:
    ret = []
    for item in results:
        target_rgba = load_hdr_rgba(item['target_image'], downsize_factor=4)
        if item['output_image'] is None:
            input_rgb_ldr = np.ones((512, 512, 3), dtype=np.float32)
            input_rgb_hdr = np.ones((512, 512, 3), dtype=np.float32)
        elif item['output_image'].endswith('.exr'):
            input_rgb_hdr = load_rgb_exr(item['output_image'])
            input_rgb_ldr = rgb_to_srgb(input_rgb_hdr)
        elif item['output_image'].endswith('.png'):
            input_rgb_ldr = load_rgb_png(item['output_image'])
            input_rgb_hdr = srgb_to_rgb(input_rgb_ldr)
        else:
            raise NotImplementedError(item['output_image'])
        ret.append(compute_similarity(input_rgb_hdr, input_rgb_ldr, target_rgba[:, :, :3], target_rgba[:, :, 3]))

    ret = list_of_dicts__to__dict_of_lists(ret)
    # print(ret)
    if LPIPS is not None:
        for k, v in ret.items():
            if np.isnan(v).any():
                logger.error(f'NAN in {k}, {v}, {np.asarray(results)[np.isnan(v)]}')
                import ipdb; ipdb.set_trace()
    ret = {k: np.mean(v) for k, v in ret.items()}
    return ret


def compute_metrics_geometry(results: List) -> Dict:
    ret = []
    for item in results:
        ret.append(dict())
        target_normal = cv2_downsize(np.load(item['target_normal']), downsize_factor=4)
        if item['output_normal'] is not None:
            input_normal = load_rgb_exr(item['output_normal'])
        else:
            input_normal = np.zeros((512, 512, 3), dtype=np.float32)
        mask = load_mask_png(Path(item['target_normal']).parent.parent.parent / 'blender_format_LDR/test_mask' / Path(item['target_normal']).with_suffix('.png').name, downsize_factor=4).astype(np.float32)
        ret[-1].update({
            'normal_angle': calc_normal_distance(input_normal, target_normal, mask),
        })
        target_depth = cv2_downsize(np.load(item['target_depth']), downsize_factor=4)
        if item['output_depth'] is not None:
            input_depth = pyexr.open(item['output_depth']).get().squeeze()
        else:
            input_depth = np.ones((512, 512))
        ret[-1].update({
            'depth_mse': calc_depth_distance(input_depth, target_depth, mask),
        })
    ret = list_of_dicts__to__dict_of_lists(ret)
    for k, v in ret.items():
        if np.isnan(v).any():
            logger.error(f'NAN in {k}, {v}, {np.asarray(results)[np.isnan(v)]}')
            # import ipdb; ipdb.set_trace()  # FIXME for numbers in the paper this SHOULD NOT HAPPEN
            # ret[k] = np.asarray(v)[~np.isnan(v)]
    ret = {k: np.mean(v) for k, v in ret.items()}
    return ret


def load_latest_leaderboard() -> Dict:
    if len(glob.glob(os.path.join(LEADERBOARD_DIR, '*.json'))) == 0:
        os.makedirs(LEADERBOARD_DIR, exist_ok=True)
        return dict()
    path = sorted(glob.glob(os.path.join(LEADERBOARD_DIR, '*.json')), key=os.path.getmtime)[-1]
    logger.info(f'Loading from {path}')
    with open(path) as f:
        return json.load(f)


def write_new_leaderboard(data):
    timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
    path = os.path.join(LEADERBOARD_DIR, f'{timestamp}.json')
    if os.path.exists(path):
        raise RuntimeError(path)

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    with open(os.path.join(LEADERBOARD_DIR, 'latest.json'), 'w') as f:
        json.dump(data, f, indent=4)


def update_leaderboard(new_data):
    data = load_latest_leaderboard()
    for k, v in new_data.items():
        overwrite_cfg(data, k, v, check_exists=False, recursive=True)
    write_new_leaderboard(data)


def load_method_leaderboard(method):
    os.makedirs(os.path.join(LEADERBOARD_DIR, 'baselines'), exist_ok=True)
    path = os.path.join(LEADERBOARD_DIR, f'baselines/{method}.json')
    if not os.path.exists(path):
        return dict()
    logger.info(f'Loading from {path}')
    with open(path) as f:
        return json.load(f)


def write_method_leaderboard(method, data):
    with open(os.path.join(LEADERBOARD_DIR, f'baselines/{method}.json'), 'w') as f:
        json.dump(data, f, indent=4)


def update_method_leaderboard(method, new_data):
    data = load_method_leaderboard(method)
    for k, v in new_data.items():
        overwrite_cfg(data, k, v, check_exists=False, recursive=True)
    write_method_leaderboard(method, data)


def compute_metrics(method):
    logger.info(f'Computing metrics for {method}')
    pipeline = build_from_config(f'imageint.pipelines.{method}.Pipeline')()

    info = dict()
    ret_new_view = dict()
    ret_new_light = dict()
    ret_geometry = dict()
    ret_material = dict()
    for scene in sorted(METADATA['training_scenes']):
        info[scene] = dict()
        if not NO_SCORE_VIEW:
            results = pipeline.test_new_view(scene, overwrite=OVERWRITE or OVERWRITE_VIEW)  # FIXME
            ret_new_view[scene] = compute_metrics_image_similarity(results)
            print('novel view', scene, ret_new_view[scene])
            info[scene].update({'view': results})

        if not NO_SCORE_LIGHT:
            results = pipeline.test_new_light(scene, overwrite=OVERWRITE or OVERWRITE_LIGHT)  # FIXME
            ret_new_light[scene] = compute_metrics_image_similarity(results)
            print('novel light', scene, ret_new_light[scene])
            info[scene].update({'light': results})

        if not NO_SCORE_GEOMETRY:
            results = pipeline.test_geometry(scene, overwrite=OVERWRITE or OVERWRITE_GEOMETRY)
            ret_geometry[scene] = compute_metrics_geometry(results)
            print('geometry', scene, ret_geometry[scene], len(results))
            info[scene].update({'geometry': results})

        if not NO_SCORE_MATERIAL:
            results = pipeline.test_material(scene, overwrite=OVERWRITE or OVERWRITE_MATERIAL)
            ret_material[scene] = compute_metrics_material(results)
            print('material', scene, ret_material[scene], len(results))
            info[scene].update({'material': results})

    scores = {'view_all': ret_new_view, 'light_all': ret_new_light, 'geometry_all': ret_geometry, 'material_all': ret_material}

    ret_new_view = list_of_dicts__to__dict_of_lists(list(ret_new_view.values()))
    print(ret_new_view)
    ret_new_view = {k: np.mean(v) for k, v in ret_new_view.items()}

    ret_new_light = list_of_dicts__to__dict_of_lists(list(ret_new_light.values()))
    print(ret_new_light)
    ret_new_light = {k: np.mean(v) for k, v in ret_new_light.items()}

    ret_geometry = list_of_dicts__to__dict_of_lists(list(ret_geometry.values()))
    print(ret_geometry)
    ret_geometry = {k: np.mean(v) for k, v in ret_geometry.items()}

    ret_material = list_of_dicts__to__dict_of_lists(list(ret_material.values()))
    print(ret_material)
    ret_material = {k: np.mean(v) for k, v in ret_material.items()}

    scores_stats = {'view': ret_new_view, 'light': ret_new_light, 'geometry': ret_geometry, 'material': ret_material}
    if os.getenv('IMAEGINT_PSEUDO_GT') == '1':
        method = method + '_pseudo_gt'
    update_method_leaderboard(method, {'scores_stats': scores_stats, 'scores': scores, 'info': info})
    update_leaderboard({'scores': {method: {'scores_stats': scores_stats, 'scores': scores}},
                        'info': {method: info}})
    logger.info('Done!')
