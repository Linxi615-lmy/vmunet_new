import os
import numpy as np

import torch
from tqdm import tqdm
import sys
sys.setrecursionlimit(10000)


from nets.unet_training import (
    BCEPlusTversky_loss,
    CE_Loss,
    CeDice_loss,
    Dice_loss,
    Tversky_loss,
)
from utils.utils import get_lr


def _normalize_to_uint8(x, vmin=None, vmax=None):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f'Expected 2D array, got shape={x.shape}')
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if vmin is None:
        vmin = float(x.min())
    if vmax is None:
        vmax = float(x.max())

    if vmax <= vmin + 1e-8:
        x = np.zeros_like(x, dtype=np.float32)
    else:
        x = (x - float(vmin)) / (float(vmax) - float(vmin))
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def _heatmap_bgr(x, vmin=None, vmax=None):
    import cv2
    return cv2.applyColorMap(_normalize_to_uint8(x, vmin=vmin, vmax=vmax), cv2.COLORMAP_HOT)


def _binary_bgr(x):
    import cv2
    x = (np.asarray(x, dtype=np.float32) > 0.5).astype(np.uint8) * 255
    return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)


def _pick_vis_channel(target_stage, error_map=None, uncertainty_map=None, rib_channels=24):
    target_np = target_stage.detach().float().cpu().numpy()
    c = min(int(rib_channels), target_np.shape[0])
    area = target_np[:c].reshape(c, -1).sum(axis=1)
    if error_map is not None and uncertainty_map is not None:
        err_np = error_map.detach().float().cpu().numpy()[:c].reshape(c, -1).sum(axis=1)
        unc_np = uncertainty_map.detach().float().cpu().numpy()[:c].reshape(c, -1).sum(axis=1)
        score = area + 0.5 * err_np + 0.5 * unc_np
    else:
        score = area
    idx = int(np.argmax(score)) if c > 0 else 0
    return idx


def _panel_with_title(img_bgr, title, panel_hw=(420, 420), title_h=44, is_mask=False):
    import cv2
    h, w = panel_hw
    canvas = np.full((h + title_h, w, 3), 255, dtype=np.uint8)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(img_bgr, (w, h), interpolation=interp)
    canvas[title_h:, :, :] = resized
    cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def _resolve_vis_root(save_dir):
    norm_dir = os.path.basename(os.path.normpath(save_dir))
    if norm_dir == 'outputlimian':
        return save_dir
    return os.path.join(save_dir, 'outputlimian')


def _tensor_image_to_bgr(x):
    import cv2

    if x is None:
        return None

    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()
    else:
        x = np.asarray(x)

    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim != 3:
        raise ValueError(f'Unsupported image shape for visualization: {x.shape}')

    if x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x = x - x.min()
    maxv = float(x.max())
    if maxv > 1e-8:
        x = x / maxv
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def _resize_map(x, hw, is_mask=False):
    import cv2
    h, w = hw
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(np.asarray(x, dtype=np.float32), (w, h), interpolation=interp)


def _blend_heatmap_on_image(base_bgr, x, alpha=0.40, vmin=None, vmax=None):
    import cv2
    heat = _heatmap_bgr(x, vmin=vmin, vmax=vmax)
    if heat.shape[:2] != base_bgr.shape[:2]:
        heat = cv2.resize(heat, (base_bgr.shape[1], base_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(base_bgr, 1.0 - alpha, heat, alpha, 0)


def _draw_mask_contour(base_bgr, mask, color=(0, 255, 0), thickness=1):
    import cv2
    mask_u8 = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.uint8) * 255
    overlay = base_bgr.copy()
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cv2.drawContours(overlay, contours, -1, color, thickness)
    return overlay


def _build_display_mask(prob, default_thresh=0.5, low_ratio=0.02, high_ratio=0.98):
    import cv2

    prob = np.asarray(prob, dtype=np.float32)
    mask_std = (prob > float(default_thresh)).astype(np.float32)
    fg_ratio = float(mask_std.mean())
    display_mask = mask_std.copy()
    display_method = f'threshold@{default_thresh:.2f}'

    if fg_ratio <= low_ratio or fg_ratio >= high_ratio:
        prob_u8 = (prob.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        otsu_thr, otsu_mask = cv2.threshold(prob_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_mask = (otsu_mask > 0).astype(np.float32)
        otsu_ratio = float(otsu_mask.mean())
        if low_ratio < otsu_ratio < high_ratio:
            display_mask = otsu_mask
            display_method = f'otsu@{otsu_thr / 255.0:.4f}'
        else:
            quantile_thr = float(np.quantile(prob, 0.995))
            if quantile_thr <= float(prob.min()) + 1e-6:
                quantile_thr = float(np.quantile(prob, 0.95))
            quantile_mask = (prob >= quantile_thr).astype(np.float32)
            quantile_ratio = float(quantile_mask.mean())
            if low_ratio < quantile_ratio < high_ratio:
                display_mask = quantile_mask
                display_method = f'quantile@{quantile_thr:.4f}'
            else:
                display_method = f'threshold@{default_thresh:.2f}(degenerate)'

    return mask_std, display_mask, {
        'fg_ratio': fg_ratio,
        'display_method': display_method,
    }


def _prepare_prediction_bundle(logits_ch, target_ch, input_tensor=None):
    gt = target_ch.detach().float().cpu().numpy()
    prob = torch.sigmoid(logits_ch).detach().float().cpu().numpy()

    base_bgr = _tensor_image_to_bgr(input_tensor)
    if base_bgr is None:
        vis_hw = gt.shape
        base_bgr = _binary_bgr(np.zeros(vis_hw, dtype=np.float32))
    else:
        vis_hw = base_bgr.shape[:2]

    gt_vis = _resize_map(gt, vis_hw, is_mask=True)
    prob_vis = _resize_map(prob, vis_hw, is_mask=False)
    mask_std, mask_disp, mask_info = _build_display_mask(prob_vis)

    # 原图 + 仅渲染红色GT真实轮廓(不带任何填充)
    gt_overlay = _draw_mask_contour(base_bgr, gt_vis, color=(0, 0, 255), thickness=1)
    # 原图 + 热力涂色图 (保留此图供需要热力的部分使用)
    prob_overlay = _blend_heatmap_on_image(base_bgr, prob_vis, alpha=0.40, vmin=0.0, vmax=1.0)
    # 彻底剥离热力图的填充涂色！仅在原图上极其清晰地画出绿色预测轮廓
    mask_overlay = _draw_mask_contour(base_bgr, mask_disp, color=(0, 255, 0), thickness=1)
    # 新增：原图 + 红色GT边界 + 绿色预测边界 (不涂色，同时呈现)，用于纯边界对比展示！
    both_overlay = _draw_mask_contour(gt_overlay, mask_disp, color=(0, 255, 0), thickness=1)

    return {
        'gt': gt,
        'prob': prob,
        'base_bgr': base_bgr,
        'vis_hw': vis_hw,
        'gt_vis': gt_vis,
        'prob_vis': prob_vis,
        'mask_std': mask_std,
        'mask_disp': mask_disp,
        'mask_info': mask_info,
        'gt_overlay': gt_overlay,
        'prob_overlay': prob_overlay,
        'mask_overlay': mask_overlay,
        'both_overlay': both_overlay,
    }


def _save_uarb_stage_vis(epoch_dir, vis_ch, uarb_out, input_tensor=None):
    import cv2

    stage_dir = os.path.join(epoch_dir, 'uarb_stage')
    os.makedirs(stage_dir, exist_ok=True)

    logits_init = uarb_out['logits_init'][0].detach()
    logits_final = uarb_out['logits_final'][0].detach()
    target_stage = uarb_out['target_stage'][0].detach()

    error_map = uarb_out.get('error_map', None)
    uncertainty_map = uarb_out.get('uncertainty_map', None)
    hard_core_map = uarb_out.get('hard_core_map', None)
    boundary_band_map = uarb_out.get('boundary_band_map', None)
    gate_map_soft = uarb_out.get('gate_map_soft', None)
    delta_logits_raw = uarb_out.get('delta_logits_raw', None)
    delta_logits_bounded = uarb_out.get('delta_logits_bounded', None)
    max_residual_logit = uarb_out.get('max_residual_logit', None)

    channel_soft = uarb_out.get('channel_soft_weight', None)
    boundary_soft = uarb_out.get('boundary_soft_weight', None)
    error_soft = uarb_out.get('error_soft_weight', None)
    tri_soft = uarb_out.get('tri_soft_weight', None)

    if error_map is not None:
        error_map = error_map[0].detach()
    if uncertainty_map is not None:
        uncertainty_map = uncertainty_map[0].detach()
    if hard_core_map is not None:
        hard_core_map = hard_core_map[0].detach()
    if boundary_band_map is not None:
        boundary_band_map = boundary_band_map[0].detach()
    if gate_map_soft is not None:
        gate_map_soft = gate_map_soft[0].detach()
    if delta_logits_raw is not None:
        delta_logits_raw = delta_logits_raw[0].detach()
    if delta_logits_bounded is not None:
        delta_logits_bounded = delta_logits_bounded[0].detach()

    if channel_soft is not None:
        channel_soft = channel_soft[0].detach()
    if boundary_soft is not None:
        boundary_soft = boundary_soft[0].detach()
    if error_soft is not None:
        error_soft = error_soft[0].detach()
    if tri_soft is not None:
        tri_soft = tri_soft[0].detach()

    stage_before = _prepare_prediction_bundle(logits_init[vis_ch], target_stage[vis_ch], input_tensor=input_tensor)
    stage_after = _prepare_prediction_bundle(logits_final[vis_ch], target_stage[vis_ch], input_tensor=input_tensor)

    delta_prob = np.abs(stage_after['prob'] - stage_before['prob'])
    delta_prob_vis = _resize_map(delta_prob, stage_after['vis_hw'], is_mask=False)

    delta_logit = (logits_final[vis_ch] - logits_init[vis_ch]).float().cpu().numpy()
    delta_logit_vis = _resize_map(delta_logit, stage_after['vis_hw'], is_mask=False)

    if error_map is None:
        error_vis = np.abs(stage_after['gt'] - stage_before['prob'])
    else:
        error_vis = error_map[vis_ch].float().cpu().numpy()
    if uncertainty_map is None:
        uncertainty_vis = 4.0 * stage_before['prob'] * (1.0 - stage_before['prob'])
    else:
        uncertainty_vis = uncertainty_map[vis_ch].float().cpu().numpy()
    error_vis = _resize_map(error_vis, stage_after['vis_hw'], is_mask=False)
    uncertainty_vis = _resize_map(uncertainty_vis, stage_after['vis_hw'], is_mask=False)

    hard_core_vis = None
    if hard_core_map is not None:
        if hard_core_map.shape[0] == 1:
            hard_core_vis = hard_core_map[0].float().cpu().numpy()
        else:
            hard_core_vis = hard_core_map[vis_ch].float().cpu().numpy()
        hard_core_vis = _resize_map(hard_core_vis, stage_after['vis_hw'], is_mask=False)

    boundary_band_vis = None
    if boundary_band_map is not None:
        if boundary_band_map.shape[0] == 1:
            boundary_band_vis = boundary_band_map[0].float().cpu().numpy()
        else:
            boundary_band_vis = boundary_band_map[vis_ch].float().cpu().numpy()
        boundary_band_vis = _resize_map(boundary_band_vis, stage_after['vis_hw'], is_mask=False)

    gate_soft_vis = None
    if gate_map_soft is not None:
        if gate_map_soft.shape[0] == 1:
            gate_soft_vis = gate_map_soft[0].float().cpu().numpy()
        else:
            gate_soft_vis = gate_map_soft[vis_ch].float().cpu().numpy()
        gate_soft_vis = _resize_map(gate_soft_vis, stage_after['vis_hw'], is_mask=False)

    delta_bounded_vis = None
    if delta_logits_bounded is not None:
        delta_bounded_vis = delta_logits_bounded[vis_ch].float().cpu().numpy()
        delta_bounded_vis = _resize_map(delta_bounded_vis, stage_after['vis_hw'], is_mask=False)

    def _to_soft_vis(tensor_map):
        if tensor_map is None: return None
        if tensor_map.shape[0] == 1:
            v = tensor_map[0].float().cpu().numpy()
        else:
            v = tensor_map[vis_ch].float().cpu().numpy()
        return _resize_map(v, stage_after['vis_hw'], is_mask=False)

    channel_soft_vis = _to_soft_vis(channel_soft)
    boundary_soft_vis = _to_soft_vis(boundary_soft)
    error_soft_vis = _to_soft_vis(error_soft)
    tri_soft_vis = _to_soft_vis(tri_soft)

    # --- 单维度图像分别保存，方便论文排版和具体问题排查 ---
    # 00_input.png: 原始输入图像，作为可视化的基准背景
    cv2.imwrite(os.path.join(stage_dir, '00_input.png'), stage_after['base_bgr'])
    # 01_gt.png: 真实标签(Ground Truth)的二值掩码图
    cv2.imwrite(os.path.join(stage_dir, '01_gt.png'), _binary_bgr(stage_after['gt_vis']))
    # 02_error_region.png: UARB模块关注的误差区域热力图，颜色越暖代表该区域预测越不准
    cv2.imwrite(os.path.join(stage_dir, '02_error_region.png'), _heatmap_bgr(error_vis, vmin=0.0, vmax=1.0))
    # 03_uncertainty_region.png: 预测不确定性热力图(通常位于概率0.5的边缘)，表征网络“拿不准”的边界
    cv2.imwrite(os.path.join(stage_dir, '03_uncertainty_region.png'), _heatmap_bgr(uncertainty_vis, vmin=0.0, vmax=1.0))
    # 03_hard_core.png: 融合了误差与不确定性的初始困难核心区域 (Error * alpha + Uncertainty * beta)
    if hard_core_vis is not None:
        cv2.imwrite(os.path.join(stage_dir, '03_hard_core.png'), _heatmap_bgr(hard_core_vis, vmin=0.0, vmax=None))
    # 03_boundary_band.png: 边界带二值化掩码，定义了预测出错时的重要惩罚区域
    if boundary_band_vis is not None:
        cv2.imwrite(os.path.join(stage_dir, '03_boundary_band.png'), _binary_bgr(boundary_band_vis))
    # 04_before_prob.png: UARB前一阶段的预测概率热力图
    cv2.imwrite(os.path.join(stage_dir, '04_before_prob.png'), _heatmap_bgr(stage_before['prob_vis'], vmin=0.0, vmax=1.0))
    # 05_after_prob.png: 经过当前UARB阶段修正后的最新预测概率热力图
    cv2.imwrite(os.path.join(stage_dir, '05_after_prob.png'), _heatmap_bgr(stage_after['prob_vis'], vmin=0.0, vmax=1.0))
    # 06_delta_prob.png: 修正前后概率图的绝对差异值(|after-before|)，直观展示当前UARB阶段重点影响了哪些地方
    cv2.imwrite(os.path.join(stage_dir, '06_delta_prob.png'), _heatmap_bgr(delta_prob_vis, vmin=0.0, vmax=0.35))

    delta_logit_cap = max(float(np.max(np.abs(delta_logit_vis))), 1e-6)
    # 07_delta_logit.png: 增量残差输出(未经过sigmoid)，区分正负修改方向
    cv2.imwrite(os.path.join(stage_dir, '07_delta_logit.png'), _heatmap_bgr(delta_logit_vis, vmin=-delta_logit_cap, vmax=delta_logit_cap))
    # 08/09_mask_05: 严格按照0.5阈值对概率进行切分得到的预测掩码
    cv2.imwrite(os.path.join(stage_dir, '08_before_mask_05.png'), _binary_bgr(stage_before['mask_std']))
    cv2.imwrite(os.path.join(stage_dir, '09_after_mask_05.png'), _binary_bgr(stage_after['mask_std']))
    # 10/11_mask_display: 自适应阈值(Otsu/九分位等)切割生成的掩码，解决极度长尾数据(全黑/全白)的展示问题
    cv2.imwrite(os.path.join(stage_dir, '10_before_mask_display.png'), _binary_bgr(stage_before['mask_disp']))
    cv2.imwrite(os.path.join(stage_dir, '11_after_mask_display.png'), _binary_bgr(stage_after['mask_disp']))
    # 12/13/14_overlay: 叠加图，分别将GT、修正前预测、修正后预测的红色或绿色轮廓画在原图上
    cv2.imwrite(os.path.join(stage_dir, '12_gt_overlay.png'), stage_after['gt_overlay'])
    cv2.imwrite(os.path.join(stage_dir, '13_before_overlay.png'), stage_before['mask_overlay'])
    cv2.imwrite(os.path.join(stage_dir, '14_after_overlay.png'), stage_after['mask_overlay'])
    # 额外存储双边界对比图(无任何涂色)
    cv2.imwrite(os.path.join(stage_dir, '14b_before_both_overlay.png'), stage_before['both_overlay'])
    cv2.imwrite(os.path.join(stage_dir, '14c_after_both_overlay.png'), stage_after['both_overlay'])
    # 获取特殊的网络细粒度特征(如有)
    if gate_soft_vis is not None:
        cv2.imwrite(os.path.join(stage_dir, '15_gate_soft.png'), _heatmap_bgr(gate_soft_vis, vmin=0.0, vmax=1.0))
    if delta_bounded_vis is not None:
        vis_cap = max(float(np.max(np.abs(delta_bounded_vis))), 1e-6)
        cv2.imwrite(os.path.join(stage_dir, '16_delta_logits_bounded.png'), _heatmap_bgr(delta_bounded_vis, vmin=-vis_cap, vmax=vis_cap))

    # 新增软权重可视化
    if channel_soft_vis is not None:
        cv2.imwrite(os.path.join(stage_dir, '20_channel_soft_weight.png'), _heatmap_bgr(channel_soft_vis, vmin=1.0, vmax=None))
    if boundary_soft_vis is not None:
        cv2.imwrite(os.path.join(stage_dir, '21_boundary_soft_weight.png'), _heatmap_bgr(boundary_soft_vis, vmin=1.0, vmax=None))
    if error_soft_vis is not None:
        cv2.imwrite(os.path.join(stage_dir, '22_error_soft_weight.png'), _heatmap_bgr(error_soft_vis, vmin=1.0, vmax=None))
    if tri_soft_vis is not None:
        cv2.imwrite(os.path.join(stage_dir, '23_tri_soft_weight.png'), _heatmap_bgr(tri_soft_vis, vmin=1.0, vmax=None))

    # 新增: FN/FP 可视化 和 上采样结果 和 npz保存
    fn_mask = (stage_after['gt_vis'] > 0.5) & (stage_after['prob_vis'] < 0.5)
    fp_mask = (stage_after['gt_vis'] <= 0.5) & (stage_after['prob_vis'] >= 0.5)
    
    cv2.imwrite(os.path.join(stage_dir, '17_false_negative.png'), _binary_bgr(fn_mask.astype(np.float32)))
    cv2.imwrite(os.path.join(stage_dir, '18_false_positive.png'), _binary_bgr(fp_mask.astype(np.float32)))
    cv2.imwrite(os.path.join(stage_dir, '19_upsampling_results.png'), _heatmap_bgr(stage_after['prob_vis'], vmin=0.0, vmax=1.0))

    npz_data = {
        'base_bgr': stage_after['base_bgr'],
        'gt_vis': stage_after['gt_vis'],
        'error_vis': error_vis,
        'uncertainty_vis': uncertainty_vis,
        'before_prob': stage_before['prob_vis'],
        'after_prob': stage_after['prob_vis'],
        'delta_prob': delta_prob_vis,
        'fn_mask': fn_mask,
        'fp_mask': fp_mask,
        'upsampling_results': stage_after['prob_vis']
    }
    if hard_core_vis is not None: npz_data['hard_core'] = hard_core_vis
    if boundary_band_vis is not None: npz_data['boundary_band'] = boundary_band_vis
    if channel_soft_vis is not None: npz_data['channel_soft'] = channel_soft_vis
    if boundary_soft_vis is not None: npz_data['boundary_soft'] = boundary_soft_vis
    if error_soft_vis is not None: npz_data['error_soft'] = error_soft_vis
    if tri_soft_vis is not None: npz_data['tri_soft'] = tri_soft_vis

    np.savez_compressed(
        os.path.join(stage_dir, 'plot_data.npz'),
        **npz_data
    )

    # --- 将这几张最核心的图组合为一个 4×2 的概览大屏，便于写论文或快速浏览对比 ---
    panels = [
        _panel_with_title(stage_after['base_bgr'], 'Input'),
        _panel_with_title(stage_after['gt_overlay'], f'UARB Stage GT (ch={vis_ch})'),
        _panel_with_title(_heatmap_bgr(error_vis, vmin=0.0, vmax=1.0), 'Error Region'),
        _panel_with_title(_heatmap_bgr(uncertainty_vis, vmin=0.0, vmax=1.0), 'Uncertainty Region'),
        _panel_with_title(stage_before['both_overlay'], 'Before Bounds (GT=R, P=G)'),
        _panel_with_title(stage_after['both_overlay'], 'After Bounds (GT=R, P=G)'),
        _panel_with_title(_heatmap_bgr(delta_prob_vis, vmin=0.0, vmax=0.35), '|Stage Delta Prob|'),
        _panel_with_title(_binary_bgr(stage_after['mask_disp']), f'Stage After Mask ({stage_after["mask_info"]["display_method"]})', is_mask=True),
    ]
    row1 = np.concatenate(panels[:4], axis=1)
    row2 = np.concatenate(panels[4:], axis=1)
    canvas = np.concatenate([row1, row2], axis=0)
    cv2.imwrite(os.path.join(stage_dir, 'paper_overview.png'), canvas)
    cv2.imwrite(os.path.join(epoch_dir, 'paper_overview_uarb_stage.png'), canvas)

    with open(os.path.join(stage_dir, 'meta.txt'), 'w', encoding='utf-8') as f:
        f.write(f'visualized_channel={vis_ch}\n')
        f.write(f'init_prob_range=({stage_before["prob"].min():.6f}, {stage_before["prob"].max():.6f})\n')
        f.write(f'final_prob_range=({stage_after["prob"].min():.6f}, {stage_after["prob"].max():.6f})\n')
        f.write(f'delta_logit_range=({delta_logit.min():.6f}, {delta_logit.max():.6f})\n')
        f.write(f'delta_logit_abs_mean={np.abs(delta_logit).mean():.6f}\n')
        f.write(f'delta_logit_abs_max={np.abs(delta_logit).max():.6f}\n')
        f.write(f'gt_fg_ratio={float((stage_after["gt"] > 0.5).mean()):.6f}\n')
        f.write(f'before_fg_ratio@0.5={stage_before["mask_info"]["fg_ratio"]:.6f}\n')
        f.write(f'after_fg_ratio@0.5={stage_after["mask_info"]["fg_ratio"]:.6f}\n')
        f.write(f'before_display_method={stage_before["mask_info"]["display_method"]}\n')
        f.write(f'after_display_method={stage_after["mask_info"]["display_method"]}\n')
        if max_residual_logit is not None:
            f.write(f'max_residual_logit={float(max_residual_logit):.6f}\n')
        if delta_logits_raw is not None:
            raw_np = delta_logits_raw[vis_ch].float().cpu().numpy()
            f.write(f'delta_logits_raw_range=({raw_np.min():.6f}, {raw_np.max():.6f})\n')
        if delta_logits_bounded is not None:
            bounded_np = delta_logits_bounded[vis_ch].float().cpu().numpy()
            f.write(f'delta_logits_bounded_range=({bounded_np.min():.6f}, {bounded_np.max():.6f})\n')
        if gate_soft_vis is not None:
            f.write(f'gate_soft_range=({gate_soft_vis.min():.6f}, {gate_soft_vis.max():.6f})\n')

        if channel_soft_vis is not None: f.write(f'channel_soft_max={channel_soft_vis.max():.6f}\n')
        if boundary_soft_vis is not None: f.write(f'boundary_soft_max={boundary_soft_vis.max():.6f}\n')
        if error_soft_vis is not None: f.write(f'error_soft_max={error_soft_vis.max():.6f}\n')
        if tri_soft_vis is not None: f.write(f'tri_soft_max={tri_soft_vis.max():.6f}\n')

        if stage_before['mask_info']['display_method'].endswith('(degenerate)') or stage_after['mask_info']['display_method'].endswith('(degenerate)'):
            f.write('warning=0.5阈值下预测掩码接近全黑或全白，overview 已优先展示 overlay 结果\n')


def _save_main_logits_vis(epoch_dir, vis_ch, main_logits, target_full, input_tensor=None):
    import cv2

    main_dir = os.path.join(epoch_dir, 'main_logits')
    os.makedirs(main_dir, exist_ok=True)

    logits = main_logits[0].detach()
    target = target_full[0].detach()
    vis_ch = min(int(vis_ch), logits.shape[0] - 1, target.shape[0] - 1)

    main_bundle = _prepare_prediction_bundle(logits[vis_ch], target[vis_ch], input_tensor=input_tensor)
    main_error = np.abs(main_bundle['gt'] - main_bundle['prob'])
    main_error_vis = _resize_map(main_error, main_bundle['vis_hw'], is_mask=False)

    # --- Main Logits 作为模型的主干预测结果，提供全局的分辨基础 ---
    # 00_input.png: 基础观测图像
    cv2.imwrite(os.path.join(main_dir, '00_input.png'), main_bundle['base_bgr'])
    # 01_gt.png: 对应真值掩码
    cv2.imwrite(os.path.join(main_dir, '01_gt.png'), _binary_bgr(main_bundle['gt_vis']))
    # 02_main_prob.png: 主干网络直接输出的预测概率图
    cv2.imwrite(os.path.join(main_dir, '02_main_prob.png'), _heatmap_bgr(main_bundle['prob_vis'], vmin=0.0, vmax=1.0))
    # 03_main_error.png: 主干网络预测图与GT的绝对误差，这里的误差就是后续 UARB 需要重点解决的区域
    cv2.imwrite(os.path.join(main_dir, '03_main_error.png'), _heatmap_bgr(main_error_vis, vmin=0.0, vmax=1.0))
    # 04_main_mask_05 / 05_main_mask_display: 0.5阈值与自适应阈值切割的二值预测结果
    cv2.imwrite(os.path.join(main_dir, '04_main_mask_05.png'), _binary_bgr(main_bundle['mask_std']))
    cv2.imwrite(os.path.join(main_dir, '05_main_mask_display.png'), _binary_bgr(main_bundle['mask_disp']))
    # 06_gt_overlay / 07_main_overlay: 将真值/预测的判定边界画在原图上的直观展示
    cv2.imwrite(os.path.join(main_dir, '06_gt_overlay.png'), main_bundle['gt_overlay'])
    cv2.imwrite(os.path.join(main_dir, '07_main_overlay.png'), main_bundle['mask_overlay'])
    # 补充无涂色的双边界对比参考图
    cv2.imwrite(os.path.join(main_dir, '08_main_both_overlay.png'), main_bundle['both_overlay'])

    # 新增: FN/FP 可视化 及 npz 数据存储
    fn_mask = (main_bundle['gt_vis'] > 0.5) & (main_bundle['prob_vis'] < 0.5)
    fp_mask = (main_bundle['gt_vis'] <= 0.5) & (main_bundle['prob_vis'] >= 0.5)

    cv2.imwrite(os.path.join(main_dir, '09_false_negative.png'), _binary_bgr(fn_mask.astype(np.float32)))
    cv2.imwrite(os.path.join(main_dir, '10_false_positive.png'), _binary_bgr(fp_mask.astype(np.float32)))

    np.savez_compressed(
        os.path.join(main_dir, 'plot_data.npz'),
        base_bgr=main_bundle['base_bgr'],
        gt_vis=main_bundle['gt_vis'],
        prob_vis=main_bundle['prob_vis'],
        error_vis=main_error_vis,
        fn_mask=fn_mask,
        fp_mask=fp_mask
    )

    # --- 拼接 Main 的可视化总结果：主要反映 UARB 介入前的全局分割质量 ---
    panels = [
        _panel_with_title(main_bundle['base_bgr'], 'Input'),
        _panel_with_title(main_bundle['gt_overlay'], f'GT Overlay (ch={vis_ch})'),
        _panel_with_title(_heatmap_bgr(main_bundle['prob_vis'], vmin=0.0, vmax=1.0), 'Main Prob'),
        _panel_with_title(_heatmap_bgr(main_error_vis, vmin=0.0, vmax=1.0), 'Main Error'),
        _panel_with_title(main_bundle['prob_overlay'], 'Main Prob Overlay'),
        _panel_with_title(main_bundle['both_overlay'], 'Main Bounds (GT=R, P=G)'),
        _panel_with_title(_binary_bgr(main_bundle['mask_std']), 'Main Mask (@0.50)', is_mask=True),
        _panel_with_title(_binary_bgr(main_bundle['mask_disp']), f'Main Mask ({main_bundle["mask_info"]["display_method"]})', is_mask=True),
    ]
    row1 = np.concatenate(panels[:4], axis=1)
    row2 = np.concatenate(panels[4:], axis=1)
    canvas = np.concatenate([row1, row2], axis=0)
    cv2.imwrite(os.path.join(main_dir, 'paper_overview.png'), canvas)
    cv2.imwrite(os.path.join(epoch_dir, 'paper_overview_main_logits.png'), canvas)

    with open(os.path.join(main_dir, 'meta.txt'), 'w', encoding='utf-8') as f:
        f.write(f'visualized_channel={vis_ch}\n')
        f.write(f'main_prob_range=({main_bundle["prob"].min():.6f}, {main_bundle["prob"].max():.6f})\n')
        f.write(f'gt_fg_ratio={float((main_bundle["gt"] > 0.5).mean()):.6f}\n')
        f.write(f'main_fg_ratio@0.5={main_bundle["mask_info"]["fg_ratio"]:.6f}\n')
        f.write(f'main_display_method={main_bundle["mask_info"]["display_method"]}\n')
        if main_bundle['mask_info']['display_method'].endswith('(degenerate)'):
            f.write('warning=0.5阈值下 main mask 接近全黑或全白，overview 已优先展示 overlay 结果\n')


def _save_compare_overview(epoch_dir, vis_ch, main_bundle, stage_bundle_before, stage_bundle_after, delta_prob_vis):
    import cv2

    # --- 横向对比全览概览 (Main vs UARB Before vs UARB After) ---
    # 这张图是最直观的消融实验图像，直接证明了 UARB 对于主干网络的补充改进能力
    panels = [
        _panel_with_title(main_bundle['base_bgr'], 'Input'), # 1. 原图
        _panel_with_title(main_bundle['gt_overlay'], f'GT Overlay (ch={vis_ch})'), # 2. 原图 + 内部真实红线标注
        _panel_with_title(main_bundle['both_overlay'], 'Main Bounds (GT=R, P=G)'), # 3. 原图 + 主干纯边界对比
        _panel_with_title(_binary_bgr(main_bundle['mask_disp']), f'Main Mask ({main_bundle["mask_info"]["display_method"]})', is_mask=True), # 4. 主干二值掩码
        _panel_with_title(stage_bundle_before['both_overlay'], 'Before Bounds (GT=R, P=G)'), # 5. 原图 + UARB修正前(红+绿双轮廓无涂色)
        _panel_with_title(stage_bundle_after['both_overlay'], 'After Bounds (GT=R, P=G)'), # 6. 原图 + UARB修正后(边界贴合度可见优化)
        _panel_with_title(_heatmap_bgr(delta_prob_vis, vmin=0.0, vmax=0.35), '|Stage Delta Prob|'), # 7. 热力图：究竟对哪里的像素进行了纠正
        _panel_with_title(_binary_bgr(stage_bundle_after['mask_disp']), f'Stage After Mask ({stage_bundle_after["mask_info"]["display_method"]})', is_mask=True), # 8. 最终的二值掩码
    ]
    row1 = np.concatenate(panels[:4], axis=1)
    row2 = np.concatenate(panels[4:], axis=1)
    canvas = np.concatenate([row1, row2], axis=0)
    cv2.imwrite(os.path.join(epoch_dir, 'paper_compare_main_vs_uarb.png'), canvas)


def _save_single_epoch_vis(save_dir, epoch, main_logits, main_target, uarb_out=None, rib_channels=24, input_tensor=None):
    vis_root = _resolve_vis_root(save_dir)
    os.makedirs(vis_root, exist_ok=True)
    epoch_dir = os.path.join(vis_root, f'epoch_{epoch:03d}')
    os.makedirs(epoch_dir, exist_ok=True)

    main_logits_0 = main_logits[0].detach()
    main_target_0 = main_target[0].detach()

    error_map = None
    uncertainty_map = None
    target_stage = None
    logits_init = None
    logits_final = None

    if uarb_out is not None:
        logits_init = uarb_out['logits_init'][0].detach()
        logits_final = uarb_out['logits_final'][0].detach()
        target_stage = uarb_out['target_stage'][0].detach()
        error_map = uarb_out.get('error_map', None)
        uncertainty_map = uarb_out.get('uncertainty_map', None)
        if error_map is not None:
            error_map = error_map[0].detach()
        if uncertainty_map is not None:
            uncertainty_map = uncertainty_map[0].detach()
        vis_ch = _pick_vis_channel(target_stage, error_map, uncertainty_map, rib_channels=rib_channels)
    else:
        vis_ch = _pick_vis_channel(main_target_0, rib_channels=rib_channels)

    max_ch = min(int(rib_channels), int(main_logits_0.shape[0]), int(main_target_0.shape[0]))
    if target_stage is not None:
        max_ch = min(max_ch, int(target_stage.shape[0]), int(logits_init.shape[0]), int(logits_final.shape[0]))
    if max_ch <= 0:
        return
    vis_ch = int(np.clip(vis_ch, 0, max_ch - 1))

    _save_main_logits_vis(epoch_dir, vis_ch, main_logits, main_target, input_tensor=input_tensor)

    with open(os.path.join(epoch_dir, 'meta.txt'), 'w', encoding='utf-8') as f:
        f.write(f'epoch={epoch}\n')
        f.write(f'visualized_channel={vis_ch}\n')

    if target_stage is None:
        with open(os.path.join(epoch_dir, 'meta.txt'), 'a', encoding='utf-8') as f:
            f.write('visual_sets=main_logits\n')
        return

    _save_uarb_stage_vis(epoch_dir, vis_ch, uarb_out, input_tensor=input_tensor)

    stage_bundle_before = _prepare_prediction_bundle(logits_init[vis_ch], target_stage[vis_ch], input_tensor=input_tensor)
    stage_bundle_after = _prepare_prediction_bundle(logits_final[vis_ch], target_stage[vis_ch], input_tensor=input_tensor)
    main_bundle = _prepare_prediction_bundle(main_logits_0[vis_ch], main_target_0[vis_ch], input_tensor=input_tensor)
    delta_prob_vis = np.abs(stage_bundle_after['prob_vis'] - stage_bundle_before['prob_vis'])
    _save_compare_overview(epoch_dir, vis_ch, main_bundle, stage_bundle_before, stage_bundle_after, delta_prob_vis)

    with open(os.path.join(epoch_dir, 'meta.txt'), 'a', encoding='utf-8') as f:
        f.write('visual_sets=main_logits,uarb_stage,compare\n')


def _default_uarb_cfg():
    return {
        'rib_channels': 24,
        'main_loss_name': 'bce_dice',
        'stage_loss_name': 'bce_dice',
        'main_bce_weight': 1.0,
        'main_dice_weight': 1.0,
        'stage_bce_weight': 1.0,
        'stage_dice_weight': 1.0,
        'tversky_alpha': 0.3,
        'tversky_beta': 0.7,
        'stage_init_weight': 0.0,
        'stage_loss_weights': [0.0, 0.5],
        'aux_warmup_epochs': 80,
        'aux_max_weight': 0.5,
        'weight_mode': 'tri_factor',
        'boundary_radius': 2,
        'lambda_channel': 0.35,
        'lambda_boundary': 1.50,
        'lambda_error': 0.75,
        'error_gamma': 1.50,
        'weight_cap': 6.0,
    }


def _merge_cfg(user_cfg):
    cfg = _default_uarb_cfg()
    if user_cfg is not None:
        cfg.update(user_cfg)
    return cfg


def _unpack_batch(batch):
    if batch is None:
        return None, None, None
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
        if len(batch) == 2:
            return batch[0], batch[1], None
    raise ValueError(f"Unsupported batch format: {type(batch)}")


def _resolve_loss_name(name, fallback='bce_dice'):
    if name is None:
        return fallback
    key = str(name).strip().lower()
    alias = {
        'diceloss': 'dice',
        'dice': 'dice',
        'ce': 'bce',
        'bce': 'bce',
        'cedice': 'bce_dice',
        'bcedice': 'bce_dice',
        'bce_dice': 'bce_dice',
        'bcetversky': 'bce_tversky',
        'bce_tversky': 'bce_tversky',
        'tversky': 'tversky',
    }
    return alias.get(key, fallback)


def _compute_seg_loss(logits, targets, valid_mask, pixel_weight, loss_name, cfg, stage_prefix='main',ifprint=None):
    key = _resolve_loss_name(loss_name)

    if key == 'dice':
        return Dice_loss(logits, targets, valid_mask=valid_mask, pixel_weight=pixel_weight)
    if key == 'bce':
        return CE_Loss(logits, targets, valid_mask=valid_mask, pixel_weight=pixel_weight)
    if key == 'tversky':
        return Tversky_loss(
            logits,
            targets,
            valid_mask=valid_mask,
            pixel_weight=pixel_weight,
            alpha=cfg.get('tversky_alpha', 0.3),
            beta=cfg.get('tversky_beta', 0.7),
        )
    if key == 'bce_tversky':
        bce_weight = cfg.get(f'{stage_prefix}_bce_weight', 1.0)
        tv_weight = cfg.get(f'{stage_prefix}_dice_weight', 1.0)
        return BCEPlusTversky_loss(
            logits,
            targets,
            valid_mask=valid_mask,
            pixel_weight=pixel_weight,
            bce_weight=bce_weight,
            tversky_weight=tv_weight,
            tversky_alpha=cfg.get('tversky_alpha', 0.3),
            tversky_beta=cfg.get('tversky_beta', 0.7),
        )

    bce_weight = cfg.get(f'{stage_prefix}_bce_weight', 1.0)
    dice_weight = cfg.get(f'{stage_prefix}_dice_weight', 1.0)
    return CeDice_loss(
        logits,
        targets,
        valid_mask=valid_mask,
        pixel_weight=pixel_weight,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        ifprint=ifprint,
    )


def _aux_schedule(epoch, cfg):
    max_weight = float(cfg.get('aux_max_weight', 1.0))
    warmup_epochs = int(cfg.get('aux_warmup_epochs', 0))
    if warmup_epochs <= 0:
        return max_weight
    progress = min(float(epoch + 1) / float(warmup_epochs), 1.0)
    return max_weight * progress


def _select_stage_weight_map(uarb_out, weight_mode):
    weight_mode = str(weight_mode).lower()

    if weight_mode == 'none':
        return None
    if weight_mode in ('pixel', 'boundary'):
        return uarb_out.get('boundary_soft_weight', uarb_out.get('pixel_soft_weight', None))
    if weight_mode == 'channel':
        return uarb_out.get('channel_soft_weight', None)
    if weight_mode == 'error':
        return uarb_out.get('error_soft_weight', None)
    if weight_mode in ('tri_factor', 'triple', 'combined', 'hybrid'):
        return uarb_out.get('tri_soft_weight', uarb_out.get('hybrid_soft_weight', None))

    raise ValueError(f"Unsupported weight_mode: {weight_mode}")


def save_train_visualization(model, vis_batch, epoch, save_dir, uarb_cfg=None):
    cfg = _merge_cfg(uarb_cfg)
    imgs_x, pngs, valid_masks = _unpack_batch(vis_batch)
    if imgs_x is None or pngs is None or imgs_x.numel() == 0:
        return

    was_training = model.training
    model.eval()

    try:
        device = next(model.parameters()).device
        with torch.no_grad():
            imgs_x = imgs_x.to(device, non_blocking=True)
            pngs = pngs.to(device, non_blocking=True)
            if valid_masks is not None:
                valid_masks = valid_masks.to(device, non_blocking=True)

            outputs = model(
                imgs_x,
                epoch=epoch,
                iter=0,
                target=pngs,
                valid_mask=valid_masks,
                uarb_cfg=cfg,
                force_uarb=True,
            )

            if not isinstance(outputs, tuple):
                return

            main_logits, uarb_outputs = outputs
            uarb_out = uarb_outputs[-1] if len(uarb_outputs) > 0 else None
            _save_single_epoch_vis(
                save_dir=save_dir,
                epoch=epoch,
                main_logits=main_logits,
                main_target=pngs,
                uarb_out=uarb_out,
                rib_channels=cfg.get('rib_channels', 24),
                input_tensor=imgs_x[0],
            )
    finally:
        if was_training:
            model.train()


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, loss_fuc, num_classes, save_dir, no_improve_count, uarb_cfg=None):
    cfg = _merge_cfg(uarb_cfg)
    total_loss = 0.0
    total_loss_main = 0.0
    total_loss_uarb = 0.0
    val_loss = 0.0
    train_steps = 0
    val_steps = 0

    pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        imgs_x, pngs, valid_masks = _unpack_batch(batch)
        if imgs_x is None or pngs is None or imgs_x.numel() == 0:
            pbar.update(1)
            continue

        with torch.no_grad():
            imgs_x = imgs_x.cuda(non_blocking=True)
            pngs = pngs.cuda(non_blocking=True)
            if valid_masks is not None:
                valid_masks = valid_masks.cuda(non_blocking=True)

        optimizer.zero_grad()

        main_logits, uarb_outputs = model_train(
            imgs_x,
            epoch=epoch,
            iter=iteration,
            target=pngs,
            valid_mask=valid_masks,
            uarb_cfg=cfg,
        )

        main_loss_name = cfg.get('main_loss_name', loss_fuc)
        loss_main = _compute_seg_loss(
            main_logits,
            pngs,
            valid_masks,
            pixel_weight=None,
            loss_name=main_loss_name,
            cfg=cfg,
            stage_prefix='main',
        )

        aux_scale = _aux_schedule(epoch, cfg)
        loss_uarb_total = torch.zeros(1, device=main_logits.device, dtype=main_logits.dtype).squeeze(0)
        weight_mode = cfg.get('weight_mode', 'hybrid')
        rib_channels = int(cfg.get('rib_channels', 24))

        for uarb_out in uarb_outputs:
            pred_stage = uarb_out['logits_final']
            target_stage = uarb_out['target_stage']
            valid_stage = uarb_out.get('valid_mask_stage', valid_masks)
            stage_scale = float(uarb_out.get('stage_loss_weight', 1.0))
            if stage_scale <= 0.0:
                continue

            stage_weight_map = _select_stage_weight_map(uarb_out, weight_mode)

            rib_c = min(rib_channels, pred_stage.shape[1])
            pred_stage = pred_stage[:, :rib_c, :, :]
            target_stage = target_stage[:, :rib_c, :, :]
            if valid_stage is not None:
                valid_stage = valid_stage[:, :rib_c]
            if stage_weight_map is not None and stage_weight_map.shape[1] not in (1, rib_c):
                stage_weight_map = stage_weight_map[:, :rib_c, :, :]

            stage_loss = _compute_seg_loss(
                pred_stage,
                target_stage,
                valid_stage,
                pixel_weight=stage_weight_map,
                loss_name=cfg.get('stage_loss_name', 'bce_dice'),
                cfg=cfg,
                stage_prefix='stage',
                ifprint=True,
            )

            loss_uarb_total = loss_uarb_total + stage_scale * stage_loss

        loss = loss_main + loss_uarb_total
        loss.backward()
        optimizer.step()

        train_steps += 1
        total_loss += float(loss.item())
        total_loss_main += float(loss_main.item())
        total_loss_uarb += float(loss_uarb_total.item())

        denom = max(train_steps, 1)
        pbar.set_postfix(**{
            'total_loss': total_loss / denom,
            'main_loss': total_loss_main / denom,
            'uarb_loss': total_loss_uarb / denom,
            'aux_scale': aux_scale,
            'lr': get_lr(optimizer),
        })
        pbar.update(1)
    pbar.close()

    print('Finish Train')
    print('Start Validation')
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        imgs_x, pngs, valid_masks = _unpack_batch(batch)
        if imgs_x is None or pngs is None or imgs_x.numel() == 0:
            pbar.update(1)
            continue

        with torch.no_grad():
            imgs_x = imgs_x.cuda(non_blocking=True)
            pngs = pngs.cuda(non_blocking=True)
            if valid_masks is not None:
                valid_masks = valid_masks.cuda(non_blocking=True)

            main_logits = model_train(imgs_x)
            loss = _compute_seg_loss(
                main_logits,
                pngs,
                valid_masks,
                pixel_weight=None,
                loss_name=cfg.get('main_loss_name', loss_fuc),
                cfg=cfg,
                stage_prefix='main',
            )
            val_loss += float(loss.item())
            val_steps += 1

        denom = max(val_steps, 1)
        pbar.set_postfix(**{'val_loss': val_loss / denom, 'lr': get_lr(optimizer)})
        pbar.update(1)
    pbar.close()

    avg_train_loss = total_loss / max(train_steps, 1)
    avg_train_main = total_loss_main / max(train_steps, 1)
    avg_train_uarb = total_loss_uarb / max(train_steps, 1)
    avg_val_loss = val_loss / max(val_steps, 1)

    print('Finish Validation')
    loss_history.append_loss(epoch + 1, avg_train_loss, avg_val_loss)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Main: %.3f || UARB: %.3f || Val Loss: %.3f ' % (
        avg_train_loss,
        avg_train_main,
        avg_train_uarb,
        avg_val_loss,
    ))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model.state_dict(), os.path.join(save_dir, 'last_epoch_weights.pth'))

    if len(loss_history.val_loss) <= 1 or avg_val_loss <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_epoch_weights.pth'))
        no_improve_count = 0
    else:
        no_improve_count += 1

    return no_improve_count
