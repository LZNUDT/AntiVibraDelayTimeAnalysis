import csv
import os
import argparse
import numpy as np
from analyze_delay import compute_delay_ms, median_smooth, moving_average, sliding_windows


def load_trajectories(traj_csv):
    times = []
    gx, gy, rx, ry = [], [], [], []
    with open(traj_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time_s']))
            def parse_val(k):
                v = row[k].strip()
                if v == '':
                    return np.nan
                try:
                    return float(v)
                except:
                    return np.nan
            gx.append(parse_val('green_x'))
            gy.append(parse_val('green_y'))
            rx.append(parse_val('red_x'))
            ry.append(parse_val('red_y'))
    times = np.array(times, dtype=float)
    gx = np.array(gx, dtype=float)
    gy = np.array(gy, dtype=float)
    rx = np.array(rx, dtype=float)
    ry = np.array(ry, dtype=float)
    # 估计FPS
    if len(times) >= 2:
        dt = np.diff(times)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        fps = 1.0 / np.median(dt) if dt.size else 200.0
    else:
        fps = 200.0
    return times, gx, gy, rx, ry, fps


def compute_metric(times, gx, gy, rx, ry, fps, axis, xcorr_window_sec, max_lag_sec, detrend_ma_sec, smooth_kernel):
    # 平滑
    gx = median_smooth(gx, smooth_kernel)
    gy = median_smooth(gy, smooth_kernel)
    rx = median_smooth(rx, smooth_kernel)
    ry = median_smooth(ry, smooth_kernel)

    # 去趋势
    if detrend_ma_sec and detrend_ma_sec > 0:
        k = int(max(1, round(detrend_ma_sec * fps)))
        if k > 1:
            gx = gx - moving_average(gx, k)
            gy = gy - moving_average(gy, k)
            rx = rx - moving_average(rx, k)
            ry = ry - moving_average(ry, k)

    # 轴选择
    axis_used = axis
    if axis == 'x':
        sig_green = gx; sig_red = rx
    elif axis == 'y':
        sig_green = gy; sig_red = ry
    else:
        # auto: 先对比两轴的最大相关
        mask_x = (~np.isnan(rx)) | (~np.isnan(gx))
        mask_y = (~np.isnan(ry)) | (~np.isnan(gy))
        _, _, corrs_x, _ = compute_delay_ms(rx[mask_x], gx[mask_x], fps, min(max_lag_sec, 0.2))
        _, _, corrs_y, _ = compute_delay_ms(ry[mask_y], gy[mask_y], fps, min(max_lag_sec, 0.2))
        max_x = np.nanmax(corrs_x) if corrs_x.size else -np.inf
        max_y = np.nanmax(corrs_y) if corrs_y.size else -np.inf
        if max_x > max_y:
            sig_green = gx; sig_red = rx; axis_used = 'x'
        else:
            sig_green = gy; sig_red = ry; axis_used = 'y'

    # 滑动窗口
    n = len(times)
    win_len = int(max(1, round(xcorr_window_sec * fps)))
    step = max(1, win_len // 4)
    delays_ms = []
    for s, e in sliding_windows(n, win_len, step):
        d_ms, _, _, _ = compute_delay_ms(sig_ref=sig_red[s:e], sig_resp=sig_green[s:e], fps=fps, max_lag_sec=max_lag_sec)
        delays_ms.append(d_ms)
    delays_ms = np.array(delays_ms, dtype=float)
    avg_delay = float(np.nanmean(delays_ms)) if delays_ms.size else np.nan
    max_delay = float(np.nanmax(delays_ms)) if delays_ms.size else np.nan
    std_delay = float(np.nanstd(delays_ms)) if delays_ms.size else np.nan
    return axis_used, avg_delay, max_delay, std_delay


def main():
    parser = argparse.ArgumentParser(description='Parameter sweep for delay analysis using existing trajectories.csv')
    parser.add_argument('--traj', type=str, default=None, help='Path to trajectories.csv (default: analysis_output/trajectories.csv next to this script)')
    parser.add_argument('--out', type=str, default=None, help='Output CSV path (default: analysis_output/param_sweep.csv)')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    traj_path = args.traj or os.path.join(base_dir, 'analysis_output', 'trajectories.csv')
    out_csv = args.out or os.path.join(base_dir, 'analysis_output', 'param_sweep.csv')

    if not os.path.isfile(traj_path):
        raise FileNotFoundError(f'Trajectories not found: {traj_path}. 请先运行 analyze_delay.py 生成 trajectories.csv')

    times, gx, gy, rx, ry, fps = load_trajectories(traj_path)

    axes = ['x', 'y', 'auto']
    max_lags = [0.03, 0.05, 0.1, 0.2]
    detrends = [0.0, 0.5, 1.0]
    windows = [2.0, 3.0, 4.0]
    smoothks = [3, 5, 7]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['axis','max_lag_sec','detrend_ma_sec','xcorr_window_sec','smooth_kernel','avg_delay_ms','max_delay_ms','std_delay_ms','fps'])
        for axis in axes:
            for max_lag in max_lags:
                for detr in detrends:
                    for win in windows:
                        for k in smoothks:
                            axis_used, avg_d, max_d, std_d = compute_metric(times, gx, gy, rx, ry, fps, axis, win, max_lag, detr, k)
                            writer.writerow([axis_used, f'{max_lag:.3f}', f'{detr:.2f}', f'{win:.2f}', k,
                                             f'{avg_d:.3f}' if np.isfinite(avg_d) else '',
                                             f'{max_d:.3f}' if np.isfinite(max_d) else '',
                                             f'{std_d:.3f}' if np.isfinite(std_d) else '', f'{fps:.3f}'])
    print(f'Param sweep saved: {out_csv}')


if __name__ == '__main__':
    main()
