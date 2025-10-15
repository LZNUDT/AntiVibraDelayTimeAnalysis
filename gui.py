import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import webbrowser
from analyze_delay import Config, HSVRange, analyze
import numpy as np
import json


class Tooltip:
    def __init__(self, widget, text, wraplength=380):
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self.tipwindow = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify=tk.LEFT,
            background="#ffffe0", relief=tk.SOLID, borderwidth=1,
            wraplength=self.wraplength
        )
        label.pack(ipadx=6, ipady=4)

    def hide_tip(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


def parse_hsv(text):
    # format: "h,s,v:h2,s2,v2" or for red two ranges separated by |
    text = text.strip()
    if not text:
        return None
    def to_range(part):
        lo, hi = part.split(":")
        lo = np.array(list(map(int, lo.split(","))))
        hi = np.array(list(map(int, hi.split(","))))
        return HSVRange(lo, hi)
    if '|' in text:
        r1s, r2s = text.split('|')
        return to_range(r1s.strip()), to_range(r2s.strip())
    return to_range(text)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("防抖延迟分析 GUI")
        self.geometry("760x560")
        self.create_widgets()

    def create_widgets(self):
        pad = 6
        frm = ttk.Frame(self)
        frm.pack(fill=tk.BOTH, expand=True, padx=pad, pady=pad)

        # Row 0: Video select
        ttk.Label(frm, text="视频文件:").grid(row=0, column=0, sticky='e')
        self.var_video = tk.StringVar()
        ent_video = ttk.Entry(frm, textvariable=self.var_video, width=70)
        ent_video.grid(row=0, column=1, sticky='we', padx=pad)
        ttk.Button(frm, text="浏览...", command=self.browse_video).grid(row=0, column=2)

        # Row 1: Output dir
        ttk.Label(frm, text="输出目录:").grid(row=1, column=0, sticky='e')
        self.var_out = tk.StringVar()
        ent_out = ttk.Entry(frm, textvariable=self.var_out, width=70)
        ent_out.grid(row=1, column=1, sticky='we', padx=pad)
        ttk.Button(frm, text="选择...", command=self.browse_outdir).grid(row=1, column=2)

        # Row 2: Axis and params
        ttk.Label(frm, text="轴:").grid(row=2, column=0, sticky='e')
        self.var_axis = tk.StringVar(value='auto')
        cmb_axis = ttk.Combobox(frm, textvariable=self.var_axis, values=['auto','x','y'], width=8, state='readonly')
        cmb_axis.grid(row=2, column=1, sticky='w')

        ttk.Label(frm, text="滑窗(秒):").grid(row=2, column=1, sticky='e', padx=(160,0))
        self.var_win = tk.StringVar(value='4.0')
        ent_win = ttk.Entry(frm, textvariable=self.var_win, width=8)
        ent_win.grid(row=2, column=1, sticky='w', padx=(240,0))

        ttk.Label(frm, text="最大滞后(秒):").grid(row=2, column=1, sticky='e', padx=(350,0))
        self.var_maxlag = tk.StringVar(value='0.1')
        ent_maxlag = ttk.Entry(frm, textvariable=self.var_maxlag, width=8)
        ent_maxlag.grid(row=2, column=1, sticky='w', padx=(450,0))

        # Row 3: detrend / filters
        ttk.Label(frm, text="去趋势MA(秒):").grid(row=3, column=0, sticky='e')
        self.var_detrend = tk.StringVar(value='0.0')
        ent_detrend = ttk.Entry(frm, textvariable=self.var_detrend, width=10)
        ent_detrend.grid(row=3, column=1, sticky='w')

        ttk.Label(frm, text="中值核: ").grid(row=3, column=1, sticky='e', padx=(160,0))
        self.var_medk = tk.StringVar(value='5')
        ent_medk = ttk.Entry(frm, textvariable=self.var_medk, width=8)
        ent_medk.grid(row=3, column=1, sticky='w', padx=(240,0))

        ttk.Label(frm, text="形态核: ").grid(row=3, column=1, sticky='e', padx=(350,0))
        self.var_morph = tk.StringVar(value='5')
        ent_morph = ttk.Entry(frm, textvariable=self.var_morph, width=8)
        ent_morph.grid(row=3, column=1, sticky='w', padx=(450,0))

        # Row 4: thresholds
        ttk.Label(frm, text="绿色HSV (lo:hi)").grid(row=4, column=0, sticky='e')
        self.var_green = tk.StringVar(value='35,60,60:85,255,255')
        ent_green = ttk.Entry(frm, textvariable=self.var_green, width=32)
        ent_green.grid(row=4, column=1, sticky='w')

        ttk.Label(frm, text="红色HSV (r1|r2)").grid(row=5, column=0, sticky='e')
        self.var_red = tk.StringVar(value='0,70,70:10,255,255|170,70,70:180,255,255')
        ent_red = ttk.Entry(frm, textvariable=self.var_red, width=32)
        ent_red.grid(row=5, column=1, sticky='w')

        # Row 6: area + debug
        ttk.Label(frm, text="最小面积: ").grid(row=6, column=0, sticky='e')
        self.var_min_area = tk.StringVar(value='100')
        ent_minarea = ttk.Entry(frm, textvariable=self.var_min_area, width=10)
        ent_minarea.grid(row=6, column=1, sticky='w')
        self.var_debug = tk.BooleanVar(value=True)
        chk_debug = ttk.Checkbutton(frm, text='保存叠加视频', variable=self.var_debug)
        chk_debug.grid(row=6, column=1, sticky='w', padx=(140,0))

        # Row 7: run/open/config
        btn_run = ttk.Button(frm, text='开始分析', command=self.run_analysis)
        btn_run.grid(row=7, column=0, pady=pad)
        ttk.Button(frm, text='打开输出目录', command=self.open_outdir).grid(row=7, column=1, sticky='w')
        ttk.Button(frm, text='保存配置', command=self.save_config).grid(row=7, column=2, sticky='w')

        # Row 7.5: batch + load config
        ttk.Button(frm, text='批量分析(文件夹)', command=self.run_batch).grid(row=8, column=0, sticky='w')
        ttk.Button(frm, text='加载配置', command=self.load_config).grid(row=8, column=1, sticky='w')

        # Row 8.5: options row (export pdf)
        self.var_export_pdf = tk.BooleanVar(value=True)
        chk_pdf = ttk.Checkbutton(frm, text='导出PDF报告', variable=self.var_export_pdf)
        chk_pdf.grid(row=8, column=2, sticky='e')

        # Row 9: progress + result
        self.pb = ttk.Progressbar(frm, mode='determinate', maximum=100)
        self.pb.grid(row=9, column=0, columnspan=3, sticky='we', pady=(pad,0))

        self.txt = tk.Text(frm, height=14)
        self.txt.grid(row=10, column=0, columnspan=3, sticky='nsew', pady=pad)
        frm.rowconfigure(10, weight=1)
        frm.columnconfigure(1, weight=1)

        # Tooltips
        Tooltip(ent_video, "选择待分析的视频文件（支持 mp4/avi/mov/mkv）。GUI 会读取视频逐帧提取红/绿目标轨迹，并计算延迟。")
        Tooltip(ent_out, "输出目录：用于保存分析结果（CSV、图表、叠加视频、summary）。若为空，默认在视频同级的 analysis_output/")
        Tooltip(cmb_axis, "分析轴：\n- x：使用水平方向像素位移\n- y：使用垂直方向像素位移（车辆Pitch通常主导Y轴）\n- auto：自动选择相关性更强的一轴")
        Tooltip(ent_win, "滑动窗口长度（秒）：互相关在该窗口内估计滞后，窗口越大越平滑，但对非平稳变化敏感度下降。")
        Tooltip(ent_maxlag, "最大滞后（秒）：互相关搜索范围。应尽可能小以避免次优峰干扰；本项目推荐 0.03~0.1s。")
        Tooltip(ent_detrend, "去趋势-移动平均（秒）：用于抑制慢变化/漂移；0 代表关闭。可尝试 0.5~1.0s。")
        Tooltip(ent_medk, "中值滤波核大小（帧）：对轨迹做抗脉冲噪声的平滑，需使用奇数，常用 3/5/7。")
        Tooltip(ent_morph, "形态学核大小（像素）：用于开闭操作，清理颜色分割噪声区域，常用 3~7。")
        Tooltip(ent_green, "绿色HSV阈值：格式 h,s,v:h2,s2,v2（下限:上限）。用于分割绿色框区域。")
        Tooltip(ent_red, "红色HSV阈值：两个范围用 | 分隔，如 r1|r2。用于分割红色贴纸区域，适配红色跨HSV两端的特性。")
        Tooltip(ent_minarea, "最小面积（像素）：小于该面积的连通域将被忽略，以抑制小噪点误检。")
        Tooltip(chk_debug, "保存叠加视频：在输出 overlay.mp4 中可视化红/绿轮廓和中心点，便于核验检测质量。")

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4;*.avi;*.mov;*.mkv"), ("All", "*.*")])
        if path:
            self.var_video.set(path)
            if not self.var_out.get():
                self.var_out.set(os.path.join(os.path.dirname(path), 'analysis_output'))

    def browse_outdir(self):
        path = filedialog.askdirectory()
        if path:
            self.var_out.set(path)

    def open_outdir(self):
        outdir = self.var_out.get().strip()
        if outdir and os.path.isdir(outdir):
            webbrowser.open(outdir)
        else:
            messagebox.showinfo("提示", "请先运行分析或选择有效输出目录。")

    def log(self, msg):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    def collect_config(self):
        cfg = Config()
        cfg.axis = self.var_axis.get()
        cfg.xcorr_window_sec = float(self.var_win.get())
        cfg.max_lag_sec = float(self.var_maxlag.get())
        cfg.detrend_ma_sec = float(self.var_detrend.get())
        cfg.smooth_kernel = int(self.var_medk.get())
        cfg.morph_kernel = int(self.var_morph.get())
        cfg.min_area = int(self.var_min_area.get())
        g = parse_hsv(self.var_green.get())
        if g is not None and not isinstance(g, tuple):
            cfg.green1 = g
        r = parse_hsv(self.var_red.get())
        if isinstance(r, tuple):
            cfg.red1, cfg.red2 = r
        return cfg

    def run_analysis(self):
        video = self.var_video.get().strip()
        if not video or not os.path.isfile(video):
            messagebox.showerror("错误", "请选择有效的视频文件")
            return
        try:
            cfg = self.collect_config()
            outdir = self.var_out.get().strip() or None
            debug = self.var_debug.get()
            export_pdf = self.var_export_pdf.get()
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return

        def update_progress(stage, p, note):
            # p: 0.0~1.0
            try:
                self.pb['value'] = max(0, min(100, int(p * 100)))
            except Exception:
                pass
            if note:
                self.log(note)

        def worker():
            try:
                self.txt.delete('1.0', tk.END)
                self.pb['value'] = 0
                def pcb(stage, p, note):
                    self.after(0, update_progress, stage, p, note)
                result = analyze(video, cfg, outdir=outdir, debug=debug, progress_cb=pcb, export_pdf=export_pdf)
                self.log("==== 分析完成 ====")
                self.log(f"Avg delay: {result['avg_delay_ms']:.2f} ms")
                self.log(f"Max delay: {result['max_delay_ms']:.2f} ms")
                self.log(f"Std delay: {result['std_delay_ms']:.2f} ms")
                self.log(f"Axis used: {result['axis_used']}")
                self.log(f"Outputs: {result['outdir']}")
            except Exception as e:
                messagebox.showerror("运行出错", str(e))
            finally:
                self.pb['value'] = 100

        threading.Thread(target=worker, daemon=True).start()

    def run_batch(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        try:
            cfg = self.collect_config()
            outdir_base = self.var_out.get().strip() or None
            debug = self.var_debug.get()
            export_pdf = self.var_export_pdf.get()
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return

        # 收集视频文件
        exts = ('.mp4', '.avi', '.mov', '.mkv')
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
        if not files:
            messagebox.showinfo("提示", "所选文件夹内未找到视频文件。")
            return

        def update_progress(stage, p, note):
            try:
                self.pb['value'] = max(0, min(100, int(p * 100)))
            except Exception:
                pass
            if note:
                self.log(note)

        def worker():
            self.txt.delete('1.0', tk.END)
            n = len(files)
            for idx, f in enumerate(files, start=1):
                self.log(f"开始分析 {idx}/{n}: {os.path.basename(f)}")
                def pcb(stage, p, note):
                    overall = ((idx - 1) + p) / n
                    self.after(0, update_progress, stage, overall, f"[{idx}/{n}] {note}")
                try:
                    analyze(f, cfg, outdir=outdir_base, debug=debug, progress_cb=pcb, export_pdf=export_pdf)
                except Exception as e:
                    self.log(f"错误: {os.path.basename(f)} -> {e}")
            self.pb['value'] = 100
            self.log("==== 批量分析完成 ====")

        threading.Thread(target=worker, daemon=True).start()

    def save_config(self):
        cfg = {
            'axis': self.var_axis.get(),
            'xcorr_window_sec': self.var_win.get(),
            'max_lag_sec': self.var_maxlag.get(),
            'detrend_ma_sec': self.var_detrend.get(),
            'smooth_kernel': self.var_medk.get(),
            'morph_kernel': self.var_morph.get(),
            'min_area': self.var_min_area.get(),
            'green': self.var_green.get(),
            'red': self.var_red.get(),
            'export_pdf': self.var_export_pdf.get(),
            'save_overlay': self.var_debug.get(),
            'outdir': self.var_out.get(),
        }
        path = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON','*.json')])
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            messagebox.showinfo('成功', f'配置已保存到\n{path}')
        except Exception as e:
            messagebox.showerror('保存失败', str(e))

    def load_config(self):
        path = filedialog.askopenfilename(filetypes=[('JSON','*.json')])
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            self.var_axis.set(cfg.get('axis','auto'))
            self.var_win.set(str(cfg.get('xcorr_window_sec','4.0')))
            self.var_maxlag.set(str(cfg.get('max_lag_sec','0.1')))
            self.var_detrend.set(str(cfg.get('detrend_ma_sec','0.0')))
            self.var_medk.set(str(cfg.get('smooth_kernel','5')))
            self.var_morph.set(str(cfg.get('morph_kernel','5')))
            self.var_min_area.set(str(cfg.get('min_area','100')))
            self.var_green.set(cfg.get('green','35,60,60:85,255,255'))
            self.var_red.set(cfg.get('red','0,70,70:10,255,255|170,70,70:180,255,255'))
            self.var_export_pdf.set(bool(cfg.get('export_pdf', True)))
            self.var_debug.set(bool(cfg.get('save_overlay', True)))
            self.var_out.set(cfg.get('outdir',''))
            messagebox.showinfo('成功', '配置已加载')
        except Exception as e:
            messagebox.showerror('加载失败', str(e))


if __name__ == '__main__':
    app = App()
    app.mainloop()
