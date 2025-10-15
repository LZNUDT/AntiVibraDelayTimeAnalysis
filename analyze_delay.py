import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import argparse
from dataclasses import dataclass
import csv
from dataclasses import field


@dataclass
class HSVRange:
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class Config:
    # 默认HSV范围，可通过命令行覆盖
    green1: HSVRange = field(default_factory=lambda: HSVRange(np.array([35, 60, 60]), np.array([85, 255, 255])))
    red1: HSVRange = field(default_factory=lambda: HSVRange(np.array([0, 70, 70]), np.array([10, 255, 255])))
    red2: HSVRange = field(default_factory=lambda: HSVRange(np.array([170, 70, 70]), np.array([180, 255, 255])))
    # 形态学
    morph_kernel: int = 5
    # 平滑滤波
    smooth_kernel: int = 5  # 中值滤波窗口，奇数
    # 滑动窗口长度（秒）用于互相关
    xcorr_window_sec: float = 4.0
    # 互相关最大允许偏移（秒）
    max_lag_sec: float = 0.1
    # 最小有效轮廓面积（像素）
    min_area: int = 100
    # 轴选择：'x' | 'y' | 'auto'
    axis: str = 'auto'
    # 去趋势窗口（秒）
    detrend_ma_sec: float = 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Anti-vibration delay analysis via color tracking and cross-correlation")
    parser.add_argument("video", type=str, help="Path to input video")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: alongside video)")
    parser.add_argument("--green", type=str, default=None, help="HSV lower,upper for green e.g. 35,60,60:85,255,255")
    parser.add_argument("--red", type=str, default=None, help="HSV for red (two ranges) e.g. 0,70,70:10,255,255|170,70,70:180,255,255")
    parser.add_argument("--axis", type=str, choices=["x","y","auto"], default="auto", help="Axis to analyze: x, y, or auto (default)")
    parser.add_argument("--xcorr_window_sec", type=float, default=None, help="Sliding window size in seconds")
    parser.add_argument("--max_lag_sec", type=float, default=None, help="Max lag search range in seconds (default 0.1)")
    parser.add_argument("--detrend_ma_sec", type=float, default=0.0, help="Moving-average window in seconds for detrending (0=off)")
    parser.add_argument("--min_area", type=int, default=None, help="Min contour area in pixels")
    parser.add_argument("--debug", action="store_true", help="Save debug overlays video")
    parser.add_argument("--export_pdf", action="store_true", help="Export analysis_report.pdf into output directory")
    return parser.parse_args()


def str_to_hsv_range(s: str) -> HSVRange:
    lo, hi = s.split(":")
    lo = np.array(list(map(int, lo.split(","))))
    hi = np.array(list(map(int, hi.split(","))))
    return HSVRange(lo, hi)


def build_config(args) -> Config:
    cfg = Config()
    if args.green:
        cfg.green1 = str_to_hsv_range(args.green)
    if args.red:
        r1s, r2s = args.red.split("|")
        cfg.red1 = str_to_hsv_range(r1s)
        cfg.red2 = str_to_hsv_range(r2s)
    cfg.axis = args.axis
    if args.xcorr_window_sec:
        cfg.xcorr_window_sec = args.xcorr_window_sec
    if args.max_lag_sec:
        cfg.max_lag_sec = args.max_lag_sec
    if args.detrend_ma_sec is not None:
        cfg.detrend_ma_sec = args.detrend_ma_sec
    if args.min_area:
        cfg.min_area = args.min_area
    return cfg


def extract_center(mask: np.ndarray, min_area: int):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    # 取最大轮廓
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < min_area:
        return None, None
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None, None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), cnt


def color_segmentation(hsv_img, hsv_range: HSVRange):
    return cv2.inRange(hsv_img, hsv_range.lower, hsv_range.upper)


def red_mask(hsv_img, r1: HSVRange, r2: HSVRange):
    m1 = color_segmentation(hsv_img, r1)
    m2 = color_segmentation(hsv_img, r2)
    return cv2.bitwise_or(m1, m2)


def preprocess_mask(mask: np.ndarray, ksize: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def median_smooth(x: np.ndarray, k: int) -> np.ndarray:
    # 使用 NumPy sliding_window_view 实现中心中值滤波（支持 NaN）
    if k <= 1:
        return x
    k = k if k % 2 == 1 else k + 1
    n = len(x)
    if n == 0:
        return x
    pad = k // 2
    # 边界用最近值扩展
    x_pad = np.pad(x, (pad, pad), mode='edge')
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(x_pad, window_shape=k)
        med = np.nanmedian(windows, axis=-1)
    except Exception:
        # 退化实现：逐点计算（较慢）
        med = np.empty(n, dtype=float)
        for i in range(n):
            seg = x_pad[i:i+k]
            med[i] = np.nanmedian(seg)
    return med


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    k = int(max(1, k))
    # 端点用边界扩展
    pad = k - 1
    x_pad = np.pad(x, (pad, 0), mode='edge')
    cumsum = np.cumsum(x_pad, dtype=float)
    ma = (cumsum[k:] - cumsum[:-k]) / k
    # 对齐为居中需再次平移，但这里用后向平均，随后再用相同k进行滚动中心化
    # 为简化，改用卷积获得中心对齐
    kernel = np.ones(k) / k
    ma_center = np.convolve(x, kernel, mode='same')
    return ma_center

def sliding_windows(n_frames: int, win_len: int, step: int):
    start = 0
    while start + win_len <= n_frames:
        yield start, start + win_len
        start += step


def compute_delay_ms(sig_ref: np.ndarray, sig_resp: np.ndarray, fps: float, max_lag_sec: float):
    """
    显式扫描滞后 L（单位：帧），定义：正的 L 表示响应(绿色)相对参考(红色)滞后 L 帧。
    即对齐方式：compare ref[t] 与 resp[t+L]。

    返回：delay_ms, best_lag_frames, corr_values, lags_array
    """
    a = sig_ref.copy().astype(float)
    b = sig_resp.copy().astype(float)
    # 去均值并替换 NaN
    a -= np.nanmean(a)
    b -= np.nanmean(b)
    a = np.nan_to_num(a)
    b = np.nan_to_num(b)

    n = len(a)
    max_lag = int(max_lag_sec * fps)
    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    corrs = np.empty_like(lags, dtype=float)

    # 归一化相关系数，避免能量差导致偏置
    for i, L in enumerate(lags):
        if L >= 0:
            a_seg = a[0:n-L]
            b_seg = b[L:n]
        else:
            a_seg = a[-L:n]
            b_seg = b[0:n+L]
        # 若有效样本太少，给极低分值
        if a_seg.size < 5:
            corrs[i] = -np.inf
            continue
        denom = np.linalg.norm(a_seg) * np.linalg.norm(b_seg)
        if denom == 0:
            corrs[i] = -np.inf
        else:
            corrs[i] = float(np.dot(a_seg, b_seg) / denom)

    if not np.isfinite(corrs).any():
        return np.nan, 0, corrs, lags

    best_idx = int(np.nanargmax(corrs))
    best_lag_frames = int(lags[best_idx])
    # 亚帧插值（抛物线拟合，使用峰及其左右点）
    lag_frac = float(best_lag_frames)
    if 0 < best_idx < len(corrs) - 1:
        y1, y2, y3 = corrs[best_idx - 1], corrs[best_idx], corrs[best_idx + 1]
        denom = (y1 - 2 * y2 + y3)
        if denom != 0:
            delta = 0.5 * (y1 - y3) / denom  # in [-0.5, 0.5] ideally
            lag_frac = best_lag_frames + float(np.clip(delta, -0.5, 0.5))
    delay_ms = (lag_frac / fps) * 1000.0
    return delay_ms, lag_frac, corrs, lags


def analyze(video_path: str, cfg: Config, outdir: str = None, debug: bool = False, progress_cb=None, export_pdf: bool = False):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    outdir = outdir or os.path.join(os.path.dirname(video_path), "analysis_output")
    os.makedirs(outdir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 200.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0 else None

    times = []
    gxs, gys = [], []
    rxs, rys = [], []

    writer = None
    if debug:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(os.path.join(outdir, 'overlay.mp4'), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        gmask = color_segmentation(hsv, cfg.green1)
        gmask = preprocess_mask(gmask, cfg.morph_kernel)
        gcenter, gcnt = extract_center(gmask, cfg.min_area)

        rmask = red_mask(hsv, cfg.red1, cfg.red2)
        rmask = preprocess_mask(rmask, cfg.morph_kernel)
        rcenter, rcnt = extract_center(rmask, cfg.min_area)

        if gcenter is None:
            gxs.append(np.nan); gys.append(np.nan)
        else:
            gxs.append(gcenter[0]); gys.append(gcenter[1])
        if rcenter is None:
            rxs.append(np.nan); rys.append(np.nan)
        else:
            rxs.append(rcenter[0]); rys.append(rcenter[1])

        times.append(frame_idx / fps)

        if writer is not None:
            overlay = frame.copy()
            if gcnt is not None:
                cv2.drawContours(overlay, [gcnt], -1, (0,255,0), 2)
                if gcenter is not None:
                    cv2.circle(overlay, gcenter, 4, (0,255,0), -1)
            if rcnt is not None:
                cv2.drawContours(overlay, [rcnt], -1, (0,0,255), 2)
                if rcenter is not None:
                    cv2.circle(overlay, rcenter, 4, (0,0,255), -1)
            writer.write(overlay)

        frame_idx += 1
        # 进度：读取阶段 0%~70%
        if progress_cb is not None and total_frames:
            p = min(0.7, 0.7 * frame_idx / max(1, total_frames))
            try:
                progress_cb('reading', p, f'Reading frames {frame_idx}/{total_frames}')
            except Exception:
                pass

    cap.release()
    if writer is not None:
        writer.release()

    times = np.array(times)
    gxs = np.array(gxs); gys = np.array(gys)
    rxs = np.array(rxs); rys = np.array(rys)
    n = len(times)

    # 平滑
    gx = median_smooth(gxs, cfg.smooth_kernel)
    gy = median_smooth(gys, cfg.smooth_kernel)
    rx = median_smooth(rxs, cfg.smooth_kernel)
    ry = median_smooth(rys, cfg.smooth_kernel)

    # 可选去趋势（移动平均）
    if cfg.detrend_ma_sec and cfg.detrend_ma_sec > 0:
        k = int(max(1, round(cfg.detrend_ma_sec * fps)))
        if k > 1:
            gx = gx - moving_average(gx, k)
            gy = gy - moving_average(gy, k)
            rx = rx - moving_average(rx, k)
            ry = ry - moving_average(ry, k)

    # 轴选择
    axis_used = cfg.axis
    if cfg.axis == 'x':
        sig_green = gx; sig_red = rx
    elif cfg.axis == 'y':
        sig_green = gy; sig_red = ry
    else:
        # auto: 选择相关性更强的轴
        d_ms_x, _, corrs_x, _ = compute_delay_ms(rx[~np.isnan(rx) | ~np.isnan(gx)], gx[~np.isnan(rx) | ~np.isnan(gx)], fps, min(cfg.max_lag_sec, 0.2))
        d_ms_y, _, corrs_y, _ = compute_delay_ms(ry[~np.isnan(ry) | ~np.isnan(gy)], gy[~np.isnan(ry) | ~np.isnan(gy)], fps, min(cfg.max_lag_sec, 0.2))
        max_x = np.nanmax(corrs_x) if corrs_x.size else -np.inf
        max_y = np.nanmax(corrs_y) if corrs_y.size else -np.inf
        if max_x > max_y:
            sig_green = gx; sig_red = rx; axis_used = 'x'
        else:
            sig_green = gy; sig_red = ry; axis_used = 'y'

    # 滑动窗口
    win_len = int(cfg.xcorr_window_sec * fps)
    step = max(1, win_len // 4)

    delays_ms = []
    delays_frames = []
    win_centers_t = []

    num_windows = 0
    for _ in sliding_windows(n, win_len, step):
        num_windows += 1
    done_windows = 0
    for s, e in sliding_windows(n, win_len, step):
        d_ms, d_frames, _, _ = compute_delay_ms(sig_ref=sig_red[s:e], sig_resp=sig_green[s:e], fps=fps, max_lag_sec=cfg.max_lag_sec)
        delays_ms.append(d_ms)
        delays_frames.append(d_frames)
        win_centers_t.append((times[s] + times[e-1]) / 2.0)
        # 进度：分析阶段 70%~100%
        done_windows += 1
        if progress_cb is not None and num_windows > 0:
            p = 0.7 + 0.3 * (done_windows / num_windows)
            try:
                progress_cb('analyzing', p, f'Analyzing windows {done_windows}/{num_windows}')
            except Exception:
                pass

    delays_ms = np.array(delays_ms, dtype=float)
    delays_frames = np.array(delays_frames, dtype=float)
    win_centers_t = np.array(win_centers_t, dtype=float)

    avg_delay = float(np.nanmean(delays_ms)) if delays_ms.size else np.nan
    max_delay = float(np.nanmax(delays_ms)) if delays_ms.size else np.nan
    std_delay = float(np.nanstd(delays_ms)) if delays_ms.size else np.nan

    # 导出CSV
    traj_path = os.path.join(outdir, 'trajectories.csv')
    with open(traj_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'time_s', 'green_x', 'green_y', 'red_x', 'red_y'])
        for i in range(n):
            writer.writerow([i, f"{times[i]:.6f}",
                             '' if np.isnan(gxs[i]) else int(gxs[i]),
                             '' if np.isnan(gys[i]) else int(gys[i]),
                             '' if np.isnan(rxs[i]) else int(rxs[i]),
                             '' if np.isnan(rys[i]) else int(rys[i])])

    delay_path = os.path.join(outdir, 'delay_windows.csv')
    with open(delay_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['win_center_time_s', 'delay_ms', 'delay_frames'])
        for i in range(len(win_centers_t)):
            writer.writerow([f"{win_centers_t[i]:.6f}", f"{delays_ms[i]:.6f}", f"{delays_frames[i]:.3f}"])

    # 可视化
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(2,1,1)
    if axis_used == 'x':
        plt.plot(times, rx, 'r-', label='Red X')
        plt.plot(times, gx, 'g-', label='Green X')
        plt.title('Trajectory (X-axis)')
        ylabel = 'Pixel (X)'
    else:
        plt.plot(times, ry, 'r-', label='Red Y')
        plt.plot(times, gy, 'g-', label='Green Y')
        plt.title('Trajectory (Y-axis)')
        ylabel = 'Pixel (Y)'
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(win_centers_t, delays_ms, 'b-o', label='Delay (ms)')
    plt.axhline(avg_delay, color='k', linestyle='--', label=f'Avg: {avg_delay:.1f} ms')
    plt.xlabel('Time (s)')
    plt.ylabel('Delay (ms)')
    plt.title(f'Sliding-window delay via cross-correlation (axis={axis_used})')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(outdir, 'analysis_plots.png')
    plt.savefig(plot_path, dpi=150)

    # 可选导出 PDF 报告
    if export_pdf:
        try:
            pdf_path = os.path.join(outdir, 'analysis_report.pdf')
            with PdfPages(pdf_path) as pdf:
                # 页1：概要+同图
                pdf.savefig(fig)
                # 页2：文本摘要
                fig2 = plt.figure(figsize=(8.27, 11.69))  # A4纵向尺寸(英寸)
                fig2.clf()
                txt = fig2.text(0.1, 0.9, '防抖延迟分析报告', fontsize=16, weight='bold')
                y = 0.85
                lines = [
                    f"视频: {os.path.basename(video_path)}",
                    f"FPS: {fps:.3f}",
                    f"帧数: {n}",
                    f"轴: {axis_used}",
                    f"最大滞后(秒): {cfg.max_lag_sec:.3f}",
                    f"滑窗(秒): {cfg.xcorr_window_sec:.3f}",
                    f"去趋势MA(秒): {cfg.detrend_ma_sec:.3f}",
                    f"中值核: {cfg.smooth_kernel}",
                    f"形态核: {cfg.morph_kernel}",
                    f"最小面积: {cfg.min_area}",
                    f"平均延迟(ms): {avg_delay:.3f}",
                    f"最大延迟(ms): {max_delay:.3f}",
                    f"标准差(ms): {std_delay:.3f}",
                    f"输出目录: {outdir}",
                ]
                for line in lines:
                    fig2.text(0.1, y, line, fontsize=11)
                    y -= 0.04
                pdf.savefig(fig2)
                plt.close(fig2)

                # 页3：检测原理
                fig3 = plt.figure(figsize=(8.27, 11.69))
                fig3.clf()
                fig3.text(0.1, 0.9, '检测原理（Methodology）', fontsize=16, weight='bold')
                y = 0.85
                methodology = [
                    '1) 颜色分割（HSV）:',
                    '   - 将每帧由BGR转换为HSV空间；',
                    '   - 绿色框：使用配置的HSV阈值单段分割；',
                    '   - 红色贴纸：考虑红色跨HSV两端，使用两段阈值取并集；',
                    '   - 对分割掩膜进行开/闭运算以抑制噪声（核大小可配）。',
                    '2) 轮廓与中心提取:',
                    '   - 在每个掩膜中选取面积最大的轮廓，忽略小于“最小面积”的连通域；',
                    '   - 通过图像矩（moments）计算质心(cx, cy)作为该目标在该帧的像素坐标。',
                    '3) 轨迹预处理:',
                    '   - 对逐帧的中心坐标进行中值滤波（奇数窗口）以抑制脉冲噪声；',
                    '   - 可选移动平均去趋势，抑制慢变漂移（窗口=秒×FPS）。',
                    '4) 延迟计算（互相关+滞后扫描）:',
                    '   - 定义红色贴纸为参考信号ref，绿色框为响应信号resp；',
                    '   - 采用显式离散滞后L扫描的归一化相关系数：对齐ref[t]与resp[t+L]；',
                    '   - 正的L表示“响应(绿色)滞后参考(红色) L 帧”（口径一致性）；',
                    '   - 在允许的最大滞后范围内选取相关系数最大的位置为最优滞后。',
                    '5) 亚帧插值（抛物线拟合）:',
                    '   - 对最优峰值及其左右邻域用二次曲线拟合，得到分数帧的滞后估计；',
                    '   - 将最优滞后(帧)换算为毫秒：delay_ms = lag_frames / FPS × 1000。',
                    '6) 滑动窗口与统计:',
                    '   - 使用长度为W秒的滑动窗口、步长为W/4估计局部延迟，提升时变场景鲁棒性；',
                    '   - 汇总所有窗口的延迟，输出平均/最大/标准差等统计指标；',
                    '   - 采用Y轴（或自动选择相关性更强的轴）以匹配车辆Pitch对垂直像素的主导影响。',
                    '7) 可视化与导出:',
                    '   - 导出逐帧轨迹CSV、窗口延迟CSV；',
                    '   - 绘制红/绿轨迹与延迟曲线图；',
                    '   - 可选导出带有轮廓与中心点标注的叠加视频；',
                    '   - 本报告自动保存上述图表与关键参数、统计结果与本文检测原理。',
                ]
                for line in methodology:
                    fig3.text(0.1, y, line, fontsize=10)
                    y -= 0.035
                    if y < 0.1:
                        pdf.savefig(fig3)
                        plt.close(fig3)
                        fig3 = plt.figure(figsize=(8.27, 11.69))
                        fig3.clf()
                        y = 0.9
                pdf.savefig(fig3)
                plt.close(fig3)
        except Exception:
            pass

    plt.close(fig)

    # 总结
    with open(os.path.join(outdir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Video: {os.path.basename(video_path)}\n")
        f.write(f"FPS: {fps:.3f}\n")
        f.write(f"Frames: {n}\n")
        f.write(f"Axis used: {axis_used}\n")
        f.write(f"Max lag search (s): {cfg.max_lag_sec:.3f}\n")
        f.write(f"Window (s): {cfg.xcorr_window_sec:.3f}\n")
        f.write(f"Detrend MA (s): {cfg.detrend_ma_sec:.3f}\n")
        f.write(f"Delay avg(ms): {avg_delay:.3f}\n")
        f.write(f"Delay max(ms): {max_delay:.3f}\n")
        f.write(f"Delay std(ms): {std_delay:.3f}\n")

    return {
        'avg_delay_ms': avg_delay,
        'max_delay_ms': max_delay,
        'std_delay_ms': std_delay,
        'axis_used': axis_used,
        'outdir': outdir,
        'fps': fps,
        'n_frames': n
    }


def main():
    args = parse_args()
    cfg = build_config(args)
    result = analyze(args.video, cfg, outdir=args.outdir, debug=args.debug, export_pdf=args.export_pdf)
    print("==== Delay analysis summary ====")
    print(f"Avg delay: {result['avg_delay_ms']:.2f} ms, Max: {result['max_delay_ms']:.2f} ms, Std: {result['std_delay_ms']:.2f} ms, Axis={result['axis_used']}")
    print(f"Outputs saved in: {result['outdir']}")


if __name__ == "__main__":
    main()
