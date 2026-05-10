import os
import numpy as np
import cv2
import glob

def _heatmap_bgr(map_1d, vmin=0., vmax=1., colormap=cv2.COLORMAP_HOT):
    # normalize
    norm = np.clip((map_1d - vmin) / (vmax - vmin + 1e-5), 0, 1)
    norm_uint8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(norm_uint8, colormap)

def _binary_bgr(mask_1d, color=(255, 255, 255)):
    # boolean mask or 0-1 mask
    m = (mask_1d > 0.5).astype(np.uint8)
    bgr = np.zeros((*m.shape, 3), dtype=np.uint8)
    bgr[m == 1] = color
    return bgr

def replot_stage_folder(npz_path):
    data = np.load(npz_path)
    folder = os.path.dirname(npz_path)
    
    # Read arrays
    base_bgr = data['base_bgr']
    gt_vis = data['gt_vis']
    error_vis = data['error_vis']
    uncertainty_vis = data['uncertainty_vis']
    upsampling_results = data['upsampling_results']
    fn_mask = data['fn_mask']
    fp_mask = data['fp_mask']
    
    hard_core = data.get('hard_core', None)
    boundary_band = data.get('boundary_band', None)
    channel_soft = data.get('channel_soft', None)
    boundary_soft = data.get('boundary_soft', None)
    error_soft = data.get('error_soft', None)
    tri_soft = data.get('tri_soft', None)
    
    # You can change the colormap or color here based on user preference
    # For example, using cv2.COLORMAP_VIRIDIS or custom RGB mapping
    # cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_OCEAN, cv2.COLORMAP_BONE, etc.
    error_color = _heatmap_bgr(error_vis, vmin=0.0, vmax=1.0, colormap=cv2.COLORMAP_HOT)
    uncertainty_color = _heatmap_bgr(uncertainty_vis, vmin=0.0, vmax=1.0, colormap=cv2.COLORMAP_HOT)
    upsampling_color = _heatmap_bgr(upsampling_results, vmin=0.0, vmax=1.0, colormap=cv2.COLORMAP_HOT)
    
    if hard_core is not None:
        hard_core_color = _heatmap_bgr(hard_core, vmin=0.0, vmax=np.mean(hard_core)*3.0, colormap=cv2.COLORMAP_HOT)
        cv2.imwrite(os.path.join(folder, 'replot_hard_core.png'), hard_core_color)
    
    # Change FN / FP colors
    # Recall FN (Missing) -> Often Blue or Red
    # FP (False Alarm) -> Often Green or Yellow
    fn_color = _binary_bgr(fn_mask, color=(0, 0, 255)) # Red in BGR
    fp_color = _binary_bgr(fp_mask, color=(0, 255, 255)) # Yellow in BGR
    
    cv2.imwrite(os.path.join(folder, 'replot_error.png'), error_color)
    cv2.imwrite(os.path.join(folder, 'replot_uncertainty.png'), uncertainty_color)
    cv2.imwrite(os.path.join(folder, 'replot_upsampling_results.png'), upsampling_color)
    cv2.imwrite(os.path.join(folder, 'replot_fn.png'), fn_color)
    cv2.imwrite(os.path.join(folder, 'replot_fp.png'), fp_color)
    
    print(f"Re-plotted in {folder}: adjust colormap in replot_npz.py if needed.")

if __name__ == '__main__':
    # usage example: scan all outputlimian npz files
    npz_files = glob.glob('outputlimian/**/plot_data.npz', recursive=True)
    for f in npz_files:
        if 'uarb_stage' in f:
            replot_stage_folder(f)
