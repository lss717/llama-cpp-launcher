import os
import sys
import atexit
import ctypes
import subprocess
import threading
import time
import queue
import yaml
import platform
import re

# CRITICAL: Must call freeze_support BEFORE any imports that use multiprocessing (psutil, etc.)
if getattr(sys, 'frozen', False):
    import multiprocessing as _mp
    _mp.freeze_support()  # Prevents child processes from re-running the whole script

import customtkinter as ctk
from tkinter import filedialog, messagebox
import psutil
import cpuinfo

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception as e:
    print(f"[WARN] NVML 初始化失败: {e}")
    HAS_NVML = False

# PyInstaller onefile: sys.executable points to temp _MEI folder, use parent process instead
_kernel32 = ctypes.windll.kernel32

if getattr(sys, 'frozen', False):
    try:
        handle = _kernel32.OpenProcess(0x410 | 0x400, False, os.getppid())
        buf = ctypes.create_unicode_buffer(260)
        size = ctypes.c_ulong(ctypes.sizeof(buf))
        if handle and _kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            BASE_DIR = os.path.dirname(buf.value)
        else:
            BASE_DIR = os.getcwd()
        _kernel32.CloseHandle(handle)
    except Exception as e:
        print(f"[WARN] 获取父进程路径失败: {e}")
        BASE_DIR = os.getcwd()
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(BASE_DIR, "config.yml")

# Single-instance protection via Windows Mutex (using ctypes, no extra dependency)
_hmutex = None
try:
    _kernel32 = ctypes.windll.kernel32
    mutex_name = "Global\\llama_cpp_launcher_v1"
    _hmutex = _kernel32.CreateMutexW(None, False, mutex_name)
    if not _hmutex:
        raise OSError("CreateMutex failed")
    last_error = _kernel32.GetLastError()
    if last_error == 183:  # ERROR_ALREADY_EXISTS
        print("[!] 程序已在运行中，退出")
        sys.exit(0)
except Exception as e:
    print(f"[WARN] Mutex 创建失败 (非致命): {e}")


def _release_mutex():
    try:
        if _hmutex is not None:
            _kernel32.ReleaseMutex(_hmutex)
    except:
        pass

atexit.register(_release_mutex)


class LlamaLauncherV6(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Llama-CPP 启动器 V1.0")
        self.geometry("1300x900")
        ctk.set_appearance_mode("dark")

        self.available_gpus = self.get_system_gpus()

        # 加载配置文件中的 profile 列表，初始化所有变量
        self.config_profiles = []
        self.current_profile = ctk.StringVar(value="")
        self.var_map = {}

        self.init_vars_and_load_profiles()

        self.process = None
        self.running = False
        self.log_queue = queue.Queue()

        # Token 统计变量
        self.total_tokens = 0
        self.tokens_per_sec = "0.00"

        # 变量就绪后，再初始化 UI
        self.setup_ui()

        # 绑定追踪逻辑（在变量和 UI 都存在后执行）
        self.bind_preview_updates()
        self.update_cmd_preview()

        self.start_monitor_thread()
        self.check_log_queue()
        # Force window visible and on top (PyInstaller console mode workaround)
        self.deiconify()
        self.lift()
        self.attributes('-topmost', True)
        self.after_idle(self.attributes, '-topmost', False)

    def get_system_gpus(self):
        gpus = []
        if HAS_NVML:
            nvml_by_name = {}
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes): name = name.decode('utf-8')
                nvml_by_name[name] = i

            # 尝试用 WMI 获取 Windows 显示适配器顺序（与任务管理器一致）
            try:
                result = subprocess.run(
                    ["wmic", "path", "win32_videocontroller", "get", "name"],
                    capture_output=True, text=True, timeout=5
                )
                lines = result.stdout.strip().split("\n")[1:]
                used = set()
                for line in lines:
                    wmi_name = line.strip()
                    if not wmi_name:
                        continue
                    if wmi_name in nvml_by_name:
                        idx = nvml_by_name[wmi_name]
                        gpus.append({"index": len(gpus), "name": wmi_name, "nvml_idx": idx})
                        used.add(wmi_name)
                    else:
                        for nvml_name, nvml_idx in nvml_by_name.items():
                            if nvml_name not in used and (nvml_name in wmi_name or wmi_name in nvml_name):
                                gpus.append({"index": len(gpus), "name": nvml_name, "nvml_idx": nvml_idx})
                                used.add(nvml_name)
                                break
            except:
                pass

            for name, nvml_idx in nvml_by_name.items():
                if name not in {g["name"] for g in gpus}:
                    gpus.append({"index": len(gpus), "name": name, "nvml_idx": nvml_idx})
        return gpus

    def setup_ui(self):
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)
        self.main_container.grid_rowconfigure(4, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)

        # 配置选择栏 (左上角)
        cfg_bar = ctk.CTkFrame(self.main_container, height=35, fg_color="#1a1a1a")
        cfg_bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        cfg_bar.pack_propagate(False)

        ctk.CTkLabel(cfg_bar, text="配置:", width=50, anchor="w").pack(side="left", padx=(5, 2))

        profile_values = self.config_profiles if self.config_profiles else ["(无)"]
        self.profile_dropdown = ctk.CTkOptionMenu(cfg_bar, variable=self.current_profile, values=profile_values, width=180, command=self.on_profile_selected)
        self.profile_dropdown.pack(side="left", padx=(0, 5))

        ctk.CTkButton(cfg_bar, text="保存", width=45, fg_color="#f1c40f", command=self.save_config).pack(side="left", padx=(0, 2))
        ctk.CTkButton(cfg_bar, text="删除", width=40, fg_color="#e74c3c", command=self.delete_profile).pack(side="left", padx=(0, 5))
        ctk.CTkEntry(cfg_bar, textvariable=self.new_config_name, placeholder_text="新配置名称").pack(side="left", fill="x", expand=True, padx=2)
        ctk.CTkButton(cfg_bar, text="+新增", width=45, fg_color="#2ecc71", command=self.add_new_profile).pack(side="right")

        # --- 需求 3: 将监控部分放在程序路径上面 ---
        self.monitor_frame = ctk.CTkFrame(self.main_container, height=120, fg_color="#1a1a1a")
        self.monitor_frame.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        self.mon_labels = {}

        # 监控项渲染 (使用 Grid 布局)
        self.mon_labels["CPU"] = ctk.CTkLabel(self.monitor_frame, text="CPU: 识别中...", anchor="w")
        self.mon_labels["CPU"].grid(row=0, column=0, padx=20, pady=5, sticky="w")

        self.mon_labels["RAM"] = ctk.CTkLabel(self.monitor_frame, text="内存: 加载中...", anchor="w")
        self.mon_labels["RAM"].grid(row=0, column=1, padx=20, pady=5, sticky="w")

        # GPU 监控动态添加
        for i, g in enumerate(self.available_gpus):
            lbl = ctk.CTkLabel(self.monitor_frame, text=f"GPU{i}: 初始化...", font=("Consolas", 11), justify="left")
            lbl.grid(row=1 + i//2, column=i%2, padx=20, pady=2, sticky="w")
            self.mon_labels[f"GPU{i}"] = lbl

        # --- 原有路径与参数区 ---
        path_frame = ctk.CTkFrame(self.main_container)
        path_frame.grid(row=2, column=0, sticky="ew", pady=5)
        self.create_input(path_frame, "程序路径:", self.server_path, browse=True)

        # Main model directory + selection
        self.create_dir_input(path_frame, "模型目录:", self.model_dir,
                              browse_cmd=lambda: self.browse_dir(self.model_dir),
                              extra_btn="刷新", extra_cmd=self.refresh_models)

        # Main model + mmproj selection row
        self.model_sel_row, self.model_dropdown, self.mmproj_dropdown = self.create_model_select_row(
            path_frame, "模型选择:", self.model_name, [],
            extra_label="多模态:", extra_var=self.mmproj_name, extra_values=["(无)"], extra_width=150)

        # Draft model selection row
        self.draft_model_row = self.create_dir_input_row(path_frame, "推测模型:", self.spec_draft_model,
                                                         browse_cmd=lambda: self.browse(self.spec_draft_model))



        param_tabs = ctk.CTkTabview(self.main_container)
        param_tabs.grid(row=3, column=0, sticky="ew", pady=10)

        # Tab 1: API/基础
        tab_api = param_tabs.add("API/基础")
        row_api = ctk.CTkFrame(tab_api, fg_color="transparent")
        row_api.pack(fill="x", padx=10, pady=5)
        self.create_small_input(row_api, "地址:", self.host)
        self.create_small_input(row_api, "端口:", self.port)
        self.create_small_option(row_api, "上下文:", self.ctx_preset,
                                 ["8192", "32768", "65536", "131072", "262144", "自定义"],
                                 width=100,
                                 command=lambda c: self.ctx_custom.set(c) if c != "自定义" else None)
        self.create_small_input(row_api, "", self.ctx_custom)
        self.create_small_check(row_api, "思考模式：", self.reasoning, text="开启")

        # Tab 2: GPU/加速
        tab_gpu = param_tabs.add("GPU/加速")
        row_gpu = ctk.CTkFrame(tab_gpu, fg_color="transparent")
        row_gpu.pack(fill="x", padx=10, pady=5)
        gpu_opts = [f"{g['index']}: {g['name']}" for g in self.available_gpus]
        if len(self.available_gpus) > 1: gpu_opts.append("所有显卡 (并行)")
        self.gpu_dropdown = self.create_small_option(row_gpu, "运行设备:", self.gpu_selection, gpu_opts, command=self.sync_main_gpu)

        self.create_small_input(row_gpu, "GPU卸载层数:", self.ngl)
        self.flash_attn_dropdown = self.create_small_option(row_gpu, "Flash Attention:", self.flash_attn, ["auto", "on", "off"], width=60)

        # 多卡并行相关控件 (单卡时隐藏，显示在下一行)
        self.multi_gpu_frame = ctk.CTkFrame(tab_gpu, fg_color="transparent")
        self.main_gpu_dropdown = self.create_small_option(self.multi_gpu_frame, "主卡编号:", self.main_gpu_index,
                                                          [str(g['index']) for g in self.available_gpus])
        self.create_small_option(self.multi_gpu_frame, "拆分模式:", self.split_mode, ["layer", "none", "row"], width=60)
        self.create_small_input(self.multi_gpu_frame, "主卡配比:", self.ts_main_val)
        self.ts_display = ctk.CTkLabel(self.multi_gpu_frame, text="-> TS: --", text_color="#3498db")
        self.ts_display.pack(side="left", padx=10)

        # Tab 3: 高级
        tab_adv = param_tabs.add("高级")
        row_adv = ctk.CTkFrame(tab_adv, fg_color="transparent")
        row_adv.pack(fill="x", padx=10, pady=5)
        self.kv_dropdown_k = self.create_small_option(row_adv, "K量化：", self.kv_quant_k, self.cache_type_options)
        self.kv_dropdown_v = self.create_small_option(row_adv, "V量化：", self.kv_quant_v, self.cache_type_options)
        self.create_small_input(row_adv, "并发量:", self.n_parallel, width=70)
        self.create_small_check(row_adv, "内存映射模型:", self.mmap)
        self.create_small_check(row_adv, "性能计时：", self.perf_timer, text="开启")
        self.create_small_check(row_adv, "MoE模型：", self.is_moe, text="启用")

        # 推测解码参数
        spec_row = ctk.CTkFrame(tab_adv, fg_color="transparent")
        spec_row.pack(fill="x", padx=10, pady=2)
        self.spec_type_dropdown = self.create_small_option(spec_row, "推测类型:", self.spec_type,
                                 ["none", "draft-simple", "draft-eagle3", "draft-mtp",
                                  "ngram-simple", "ngram-map-k", "ngram-map-k4v",
                                  "ngram-mod", "ngram-cache",
                                  "suffix", "copyspec", "recycle", "dflash"],
                                 command=self.on_spec_type_changed)
        self.spec_sub_frame = ctk.CTkFrame(spec_row, fg_color="transparent")
        self.spec_sub_frame.pack(side="left")
        ctk.CTkLabel(self.spec_sub_frame, text="推测Token数:").pack(side="left", padx=(5, 2))
        self.spec_nmax_entry = ctk.CTkEntry(self.spec_sub_frame, textvariable=self.spec_draft_n_max, width=60)
        self.spec_nmax_entry.pack(side="left", padx=5)

        # DFlash专用参数 (仅选dflash时显示)
        self.dflash_frame = ctk.CTkFrame(tab_adv, fg_color="transparent")
        self.create_small_check(self.dflash_frame, "默认配置:", self.spec_dflash_default, text="启用")
        self.dflash_params_frame = ctk.CTkFrame(self.dflash_frame, fg_color="transparent")
        self.create_small_input(self.dflash_params_frame, "DFlash槽位数:", self.spec_dflash_max_slots, width=60)
        self.create_small_input(self.dflash_params_frame, "交叉上下文:", self.spec_dflash_cross_ctx, width=70)
        self.create_small_input(self.dflash_params_frame, "Draft Top-K:", self.spec_draft_top_k, width=60)
        self.create_small_input(self.dflash_params_frame, "Draft温度:", self.spec_draft_temp, width=60)
        self.dflash_params_frame.pack(side="left")

        # MoE模型专属参数 (仅MoE模型时显示)
        self.moe_frame = ctk.CTkFrame(tab_adv, fg_color="transparent")
        self.create_small_check(self.moe_frame, "CPU常驻MoE层:", self.cpu_moe, text="启用")
        self.create_small_input(self.moe_frame, "前N层CPU:", self.n_cpu_moe, width=60)

        # 额外参数输入框（独占一行，填满宽度）
        self.extra_args_frame = ctk.CTkFrame(tab_adv, fg_color="transparent")
        self.extra_args_frame.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(self.extra_args_frame, text="额外参数:", width=90, anchor="w").pack(side="left")
        ctk.CTkEntry(self.extra_args_frame, textvariable=self.extra_args).pack(side="left", fill="x", expand=True, padx=(5, 5))

        self.on_spec_type_changed(self.spec_type.get())
        self.is_moe.trace_add("write", lambda *a: self.toggle_moe_frame())
        self.spec_dflash_default.trace_add("write", lambda *a: self.toggle_dflash_params())
        self.toggle_moe_frame()
        self.toggle_dflash_params()

        # 命令预览 和 日志区
        log_box = ctk.CTkFrame(self.main_container, fg_color="transparent")
        log_box.grid(row=4, column=0, sticky="nsew", pady=(5, 2))
        log_box.grid_rowconfigure(1, weight=0)
        log_box.grid_rowconfigure(3, weight=6)
        log_box.grid_columnconfigure(0, weight=1)

        # 命令预览
        ctk.CTkLabel(log_box, text="[ 命令行预览 ]", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        self.cmd_display = ctk.CTkTextbox(
                log_box,
                font=("Consolas", 10),
                text_color="#3498db",
                height=60,
                activate_scrollbars=False
            )
        self.cmd_display.grid(row=1, column=0, sticky="nsew", pady=(2, 4))
        self.cmd_display.configure(state="disabled")

        # 日志区
        self.log_stats_var = ctk.StringVar(value="")
        ctk.CTkLabel(log_box, text="[ 实时运行日志 ]", font=("Segoe UI", 12, "bold")).grid(row=2, column=0, sticky="w")
        self.log_stats_label = ctk.CTkLabel(log_box, textvariable=self.log_stats_var, font=("Consolas", 11), text_color="#3498db")
        self.log_stats_label.grid(row=2, column=0, sticky="w", padx=(160, 0))
        self.log_display = ctk.CTkTextbox(log_box, font=("Consolas", 11), fg_color="#0d0d0d")
        self.log_display.grid(row=3, column=0, sticky="nsew", pady=(0, 5))

        # 配置日志颜色标签
        self.log_display.tag_config("info", foreground="#2ecc71")
        self.log_display.tag_config("warn", foreground="#f1c40f")
        self.log_display.tag_config("error", foreground="#e74c3c")
        self.log_display.tag_config("highlight", foreground="#3498db")

        # 4. 按钮
        btn_row = ctk.CTkFrame(self.main_container, fg_color="transparent")
        btn_row.grid(row=5, column=0, sticky="ew", pady=5)
        self.start_btn = ctk.CTkButton(btn_row, text="🚀 启动服务", fg_color="#2ecc71", command=self.start_server, height=40)
        self.start_btn.pack(side="left", expand=True, padx=5)
        self.stop_btn = ctk.CTkButton(btn_row, text="🛑 停止服务", fg_color="#e74c3c", command=self.stop_server, state="disabled", height=40)
        self.stop_btn.pack(side="left", expand=True, padx=5)

        # 绑定逻辑
        for var in [self.gpu_selection, self.main_gpu_index, self.ts_main_val]:
            var.trace_add("write", lambda *args: self.auto_calc_ts())
        self.sync_main_gpu(self.gpu_selection.get())

    def bind_preview_updates(self):
        """为所有影响命令的变量绑定更新函数"""
        vars_to_track = [
            self.server_path, self.model_dir, self.model_name,
            self.mmproj_name, self.host, self.port, self.ngl, self.ctx_custom,
            self.ts_final_str, self.kv_quant_k, self.kv_quant_v, self.reasoning,
            self.gpu_selection, self.main_gpu_index, self.n_parallel,
            self.perf_timer, self.mmap, self.flash_attn, self.split_mode,
            self.spec_type, self.spec_draft_model, self.spec_draft_n_max,
            self.spec_dflash_max_slots, self.spec_dflash_cross_ctx,
            self.spec_draft_top_k, self.spec_draft_temp, self.spec_dflash_default,
            self.cpu_moe, self.n_cpu_moe, self.is_moe,
            self.extra_args
        ]
        for var in vars_to_track:
            var.trace_add("write", lambda *args: self.update_cmd_preview())

    def refresh_models(self):
        d = self.model_dir.get().strip()
        if not d or not os.path.isdir(d):
            return
        all_files = sorted(os.listdir(d))
        model_files = [f for f in all_files if not f.startswith("mmproj") and not f.startswith(".")]
        mmproj_files = [f for f in all_files if f.startswith("mmproj")]

        current_model = self.model_name.get()
        current_mmproj = self.mmproj_name.get()

        model_opts = model_files if model_files else ["(无)"]
        self.model_dropdown.configure(values=model_opts)
        if current_model not in model_opts and model_files:
            self.model_name.set(model_files[0])
        elif current_model in model_opts:
            self.model_name.set(current_model)

        mmproj_opts = ["(无)"] + mmproj_files
        self.mmproj_dropdown.configure(values=mmproj_opts)
        if current_mmproj not in mmproj_opts and mmproj_files:
            self.mmproj_name.set(mmproj_files[0])
        elif current_mmproj in mmproj_opts:
            self.mmproj_name.set(current_mmproj)

    def get_full_model_path(self):
        d = self.model_dir.get().strip()
        n = self.model_name.get().strip()
        if d and n and n != "(无)":
            return os.path.join(d, n)
        return ""

    def get_full_mmproj_path(self):
        d = self.model_dir.get().strip()
        n = self.mmproj_name.get().strip()
        if d and n and n != "(无)":
            return os.path.join(d, n)
        return ""

    def browse_dir(self, var):
        p = filedialog.askdirectory()
        if p: var.set(p)

    def get_cuda_visible_devices(self):
        """根据GPU选择获取CUDA_VISIBLE_DEVICES值"""
        sel = self.gpu_selection.get()
        if "所有显卡" in sel:
            return ",".join([str(g['index']) for g in self.available_gpus])
        return sel.split(":")[0]

    def build_command_list(self, for_display=False):
        """构建完整的llama-server命令"""
        mg_idx = self.main_gpu_index.get() if self.main_gpu_index.get() else "0"
        quote = lambda s: f'"{s}"' if for_display else s

        sel = self.gpu_selection.get()
        is_parallel = "所有显卡" in sel

        cmd = [
            quote(self.server_path.get()),
            "--host", self.host.get(),
            "--port", self.port.get(),
            "--model", quote(self.get_full_model_path()),
            "--gpu-layers", self.ngl.get(),
        ]
        if is_parallel:
            cmd.extend([
                "--main-gpu", mg_idx,
                "--tensor-split", self.ts_final_str.get(),
                "--split-mode", self.split_mode.get(),
            ])
        cmd.extend([
            "--ctx-size", self.ctx_custom.get(),
            "--cache-type-k", self.kv_quant_k.get(),
            "--cache-type-v", self.kv_quant_v.get(),
            "--parallel", self.n_parallel.get(),
        ])
        if self.reasoning.get() == "on":
            cmd.extend(["--reasoning", "on"])
        else:
            cmd.extend(["--reasoning", "off"])

        if self.perf_timer.get() == "on":
            cmd.append("--perf")

        if self.mmap.get() != "on":
            cmd.append("--no-mmap")
        else:
            cmd.append("--mmap")

        if self.flash_attn.get() != "auto":
            cmd.extend(["--flash-attn", self.flash_attn.get()])

        mmproj_full = self.get_full_mmproj_path()
        if mmproj_full:
            cmd.extend(["--mmproj", quote(mmproj_full)])

        if self.spec_type.get() and self.spec_type.get() != "none":
            cmd.extend(["--spec-type", self.spec_type.get()])

            if self.spec_type.get() != "draft-mtp":
                spec_draft = self.spec_draft_model.get().strip()
                if spec_draft:
                    cmd.extend(["--spec-draft-model", quote(spec_draft)])

            if self.spec_draft_n_max.get():
                cmd.extend(["--spec-draft-n-max", self.spec_draft_n_max.get()])

            if self.spec_type.get() == "dflash":
                if self.spec_dflash_default.get() == "on":
                    cmd.append("--spec-dflash-default")
                else:
                    if self.spec_dflash_max_slots.get():
                        cmd.extend(["--spec-dflash-max-slots", self.spec_dflash_max_slots.get()])
                    if self.spec_dflash_cross_ctx.get():
                        cmd.extend(["--spec-dflash-cross-ctx", self.spec_dflash_cross_ctx.get()])
                    if self.spec_draft_top_k.get():
                        cmd.extend(["--spec-draft-top-k", self.spec_draft_top_k.get()])
                    if self.spec_draft_temp.get():
                        cmd.extend(["--spec-draft-temp", self.spec_draft_temp.get()])

        if self.cpu_moe.get() == "on":
            cmd.append("--cpu-moe")
        if self.n_cpu_moe.get():
            cmd.extend(["--n-cpu-moe", self.n_cpu_moe.get()])

        extra = self.extra_args.get().strip()
        if extra:
            for arg in re.split(r'\s+', extra):
                cmd.append(arg)

        if for_display:
            return " ".join(cmd)
        return cmd

    def update_cmd_preview(self):
        """更新命令行预览显示"""
        cmd_str = self.build_command_list(for_display=True)
        sel = self.gpu_selection.get()
        if "所有显卡" not in sel and "CPU" not in sel:
            cmd_str = f'set CUDA_VISIBLE_DEVICES={sel.split(":")[0]} && ' + cmd_str
        self.cmd_display.configure(state="normal")
        self.cmd_display.delete("1.0", "end")
        self.cmd_display.insert("1.0", cmd_str)
        self.cmd_display.configure(state="disabled")

    def start_monitor_thread(self):
        # 1. 使用 cpuinfo 获取真实 CPU 名称
        try:
            full_cpu_info = cpuinfo.get_cpu_info()
            cpu_name = full_cpu_info.get('brand_raw', "Unknown CPU")
        except:
            cpu_name = platform.processor()

        def monitor():
            while True:
                try:
                    # CPU 占用率监控 (移除温度获取)
                    cpu_usage = psutil.cpu_percent()
                    self.mon_labels["CPU"].configure(
                        text=f"处理器: {cpu_name} | 占用: {cpu_usage}%"
                    )

                    # 内存监控
                    ram = psutil.virtual_memory()
                    self.mon_labels["RAM"].configure(
                        text=f"内存: {ram.used/1024**3:.1f}G / {ram.total/1024**3:.1f}G ({ram.percent}%)"
                    )

                    # GPU 详细监控 (添加显卡名称显示)
                    if HAS_NVML:
                        for i, gpu_info in enumerate(self.available_gpus):
                            h = pynvml.nvmlDeviceGetHandleByIndex(gpu_info['nvml_idx'])
                            name = gpu_info['name'] # 获取初始化时记录的显卡名称

                            info = pynvml.nvmlDeviceGetMemoryInfo(h)
                            util = pynvml.nvmlDeviceGetUtilizationRates(h)
                            temp = pynvml.nvmlDeviceGetTemperature(h, 0)
                            pwr = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0

                            # 格式化显示：[GPU索引] 型号 | 负载 | 显存 | 温度 | 功耗
                            display_text = (
                                f"GPU{i} [{name}]: 核心 {util.gpu}% | "
                                f"显存 {info.used/1024**3:.1f}G/{info.total/1024**3:.1f}G | "
                                f"{temp}°C | {pwr:.1f}W"
                            )
                            self.mon_labels[f"GPU{i}"].configure(text=display_text)
                except Exception as e:
                    print(f"监控线程异常: {e}")

                time.sleep(1)

        threading.Thread(target=monitor, daemon=True).start()

    def highlight_logs(self, text):
        """解析 Token 并高亮日志"""
        self.log_display.insert("end", text)

        if "total time =" in text:
            m = re.search(r"total time =\s+[\d\.]+\s+ms\s+/\s+(\d+)\s+tokens", text)
            if m:
                self.total_tokens = int(m.group(1))
        if "eval time =" in text:
            m = re.search(r"([\d\.]+)\s+tokens per second", text)
            if m:
                self.tokens_per_sec = m.group(1)

        ctx_limit = int(self.ctx_custom.get()) if self.ctx_custom.get().isdigit() else 32768
        percent = min(100, int((self.total_tokens / ctx_limit) * 100)) if self.total_tokens else 0
        self.log_stats_var.set(f"Tokens: {self.total_tokens} | 速度: {self.tokens_per_sec} t/s | 占比: {percent}%")

        # 高亮逻辑
        last_line_idx = self.log_display.index("end-2c linestart")
        end_idx = self.log_display.index("end-1c")
        content = text.upper()
        if "ERR" in content or "FAIL" in content:
            self.log_display.tag_add("error", last_line_idx, end_idx)
        elif "WARN" in content:
            self.log_display.tag_add("warn", last_line_idx, end_idx)
        elif "LLAMA" in content:
            self.log_display.tag_add("highlight", last_line_idx, end_idx)


    def start_server(self):
        if self.running: return
        self.total_tokens = 0
        self.tokens_per_sec = "0.00"
        self.log_stats_var.set("")
        self.log_display.delete("1.0", "end")

        cmd = self.build_command_list(for_display=False)

        env = os.environ.copy()
        sel = self.gpu_selection.get()
        if "所有显卡" not in sel:
            env["CUDA_VISIBLE_DEVICES"] = sel.split(":")[0]

        def run():
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8', errors='replace', creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW, env=env)
            self.running = True
            for line in iter(self.process.stdout.readline, ""):
                self.log_queue.put(line)
            self.running = False

        threading.Thread(target=run, daemon=True).start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="enabled")

    def stop_server(self):
        if self.process and self.process.poll() is None:
            try:
                pid = self.process.pid
                proc = psutil.Process(pid)
                for child in proc.children(recursive=True):
                    child.terminate()
                proc.terminate()
                gone, alive = psutil.wait_procs([proc] + proc.children(recursive=True), timeout=3)
                for p in alive:
                    try:
                        p.kill()
                    except psutil.NoSuchProcess:
                        pass
            except Exception as e:
                print(f"终止进程异常: {e}")

            stop_msg = "\n" + "="*20 + " 服务停止成功 " + "="*20 + "\n"
            self.log_queue.put(stop_msg)

        self.running = False
        self.start_btn.configure(state="enabled")
        self.stop_btn.configure(state="disabled")

    def check_log_queue(self):
        while not self.log_queue.empty():
            self.highlight_logs(self.log_queue.get())
            self.log_display.see("end")
        self.after(100, self.check_log_queue)

    def init_vars_and_load_profiles(self):
        self.defaults = {
            "server_path": r"D:\Program Files\llama\llama-server.exe",
            "model_dir": "", "model_name": "", "mmproj_name": "",
            "host": "0.0.0.0", "port": "8080",
            "ngl": "all", "ctx": "32768", "ts_ratio": "28",
            "cache_type": "q8_0",
            "mmap": "off", "perf_timer": "off"
        }

        self.server_path = ctk.StringVar(value=self.defaults["server_path"])
        self.model_dir = ctk.StringVar()
        self.model_name = ctk.StringVar()
        self.mmproj_name = ctk.StringVar()
        self.host = ctk.StringVar(value=self.defaults["host"])
        self.port = ctk.StringVar(value=self.defaults["port"])
        self.ngl = ctk.StringVar(value=self.defaults["ngl"])
        self.ctx_custom = ctk.StringVar(value=self.defaults["ctx"])
        self.ts_main_val = ctk.StringVar(value=self.defaults["ts_ratio"])
        self.kv_quant_k = ctk.StringVar(value=self.defaults["cache_type"])
        self.kv_quant_v = ctk.StringVar(value=self.defaults["cache_type"])
        self.flash_attn = ctk.StringVar(value="auto")
        self.split_mode = ctk.StringVar(value="layer")
        self.perf_timer = ctk.StringVar(value="off")
        self.is_moe = ctk.StringVar(value="off")
        self.mmap = ctk.StringVar(value="on")
        self.ctx_preset = ctk.StringVar(value="自定义")
        self.reasoning = ctk.StringVar(value="off")
        self.cache_type_options = ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
        self.n_parallel = ctk.StringVar(value="-1")
        self.spec_type = ctk.StringVar(value="none")
        self.spec_draft_model = ctk.StringVar()
        self.spec_draft_n_max = ctk.StringVar(value="16")
        self.spec_dflash_max_slots = ctk.StringVar(value="1")
        self.spec_dflash_cross_ctx = ctk.StringVar(value="512")
        self.spec_draft_top_k = ctk.StringVar(value="1")
        self.spec_draft_temp = ctk.StringVar(value="0.0")
        self.spec_dflash_default = ctk.StringVar(value="off")
        self.cpu_moe = ctk.StringVar(value="off")
        self.n_cpu_moe = ctk.StringVar(value="")
        self.extra_args = ctk.StringVar()
        self.ts_final_str = ctk.StringVar(value="1")

        gpu_opts = [f"{g['index']}: {g['name']}" for g in self.available_gpus]
        if not gpu_opts:
            gpu_opts.append("CPU (无显卡)")
        elif len(gpu_opts) > 1:
            gpu_opts.append("所有显卡 (并行)")
        self.gpu_selection = ctk.StringVar(value=gpu_opts[-1])
        self.main_gpu_index = ctk.StringVar(value="0")

        self.var_map = {
            "server_path": self.server_path, "model_dir": self.model_dir,
            "model_name": self.model_name, "mmproj_name": self.mmproj_name,
            "host": self.host, "port": self.port, "ngl": self.ngl,
            "ctx": self.ctx_custom, "ts_ratio": self.ts_main_val,
            "cache_type_k": self.kv_quant_k, "cache_type_v": self.kv_quant_v,
            "mmap": self.mmap,
            "perf_timer": self.perf_timer, "is_moe": self.is_moe, "flash_attn": self.flash_attn,
            "split_mode": self.split_mode, "reasoning": self.reasoning,
            "gpu_selection": self.gpu_selection, "n_parallel": self.n_parallel,
            "spec_type": self.spec_type, "spec_draft_model": self.spec_draft_model,
            "spec_draft_n_max": self.spec_draft_n_max,
            "spec_dflash_max_slots": self.spec_dflash_max_slots,
            "spec_dflash_cross_ctx": self.spec_dflash_cross_ctx,
            "spec_draft_top_k": self.spec_draft_top_k,
            "spec_draft_temp": self.spec_draft_temp,
            "spec_dflash_default": self.spec_dflash_default,
            "cpu_moe": self.cpu_moe, "n_cpu_moe": self.n_cpu_moe,
            "extra_args": self.extra_args,
        }

        self.new_config_name = ctk.StringVar()
        self._load_profiles_from_file()

    def _load_profiles_from_file(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                all_cfg = yaml.safe_load(f) or {}
            profiles = {k: v for k, v in all_cfg.items() if isinstance(v, dict)}
            self.config_profiles = list(profiles.keys())
            if self.config_profiles:
                self.current_profile.set(self.config_profiles[0])
                self.apply_profile(self.config_profiles[0], profiles[self.config_profiles[0]])

    def apply_profile(self, name, cfg):
        for key, var in self.var_map.items():
            val = cfg.get(key)
            if val is not None:
                str_val = str(val) if not isinstance(val, list) else None
                if str_val and key == "perf_timer":
                    var.set("on" if str_val in ("on", "1") else "off")
                elif str_val and key == "mmap":
                    var.set("on" if str_val in ("on", "1") else "off")
                elif str_val and key == "reasoning":
                    var.set("on" if str_val in ("on", "1") else "off")
                elif str_val and key == "is_moe":
                    var.set("on" if str_val in ("on", "1") else "off")
                elif str_val is not None:
                    var.set(str_val)
        cache_opts = cfg.get("cache_type_options")
        if isinstance(cache_opts, list):
            self.cache_type_options = cache_opts
            if hasattr(self, 'kv_dropdown_k'):
                self.kv_dropdown_k.configure(values=cache_opts)
                self.kv_dropdown_v.configure(values=cache_opts)
        if hasattr(self, 'model_dropdown'):
            self.refresh_models()
            self.sync_main_gpu(self.gpu_selection.get())
            self.on_spec_type_changed(self.spec_type.get())
            self.toggle_moe_frame()

    def on_profile_selected(self, _=None):
        name = self.current_profile.get()
        if not name or name == "(无)": return
        all_cfg = {}
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                all_cfg = yaml.safe_load(f) or {}
        profiles = {k: v for k, v in all_cfg.items() if isinstance(v, dict)}
        if name in profiles:
            self.apply_profile(name, profiles[name])

    def delete_profile(self):
        name = self.current_profile.get()
        if not name or name == "(无)": return
        all_cfg = {}
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                all_cfg = yaml.safe_load(f) or {}
        profiles = {k: v for k, v in all_cfg.items() if isinstance(v, dict)}
        non_profiles = {k: v for k, v in all_cfg.items() if not isinstance(v, dict)}
        if name in profiles:
            del profiles[name]
            self.config_profiles.remove(name)
            profile_values = self.config_profiles if self.config_profiles else ["(无)"]
            self.profile_dropdown.configure(values=profile_values)
            self.current_profile.set(profile_values[0])
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(dict(non_profiles, **profiles), f, allow_unicode=True, sort_keys=False)

    def add_new_profile(self):
        name = self.new_config_name.get().strip()
        if not name:
            messagebox.showwarning("提示", "配置名称不能为空")
            return
        all_cfg = {}
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                all_cfg = yaml.safe_load(f) or {}
        profiles = {k: v for k, v in all_cfg.items() if isinstance(v, dict)}
        non_profiles = {k: v for k, v in all_cfg.items() if not isinstance(v, dict)}
        if name not in profiles:
            profiles[name] = {}
        self.config_profiles.append(name)
        self.current_profile.set(name)
        self.profile_dropdown.configure(values=self.config_profiles)

    def save_config(self):
        profile_name = self.current_profile.get()
        if not profile_name or profile_name == "(无)": return
        all_cfg = {}
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                all_cfg = yaml.safe_load(f) or {}
        profiles = {k: v for k, v in all_cfg.items() if isinstance(v, dict)}
        non_profiles = {k: v for k, v in all_cfg.items() if not isinstance(v, dict)}
        profile_data = profiles.get(profile_name, {})
        for key, var in self.var_map.items():
            profile_data[key] = var.get()
        cache_opts = self.cache_type_options
        if isinstance(cache_opts, list):
            profile_data["cache_type_options"] = cache_opts
        profiles[profile_name] = profile_data
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(dict(non_profiles, **profiles), f, allow_unicode=True, sort_keys=False)

    def sync_main_gpu(self, choice):
        if "所有显卡" not in choice:
            idx = choice.split(":")[0]; self.main_gpu_index.set(idx)
            self.main_gpu_dropdown.configure(state="disabled")
            self.multi_gpu_frame.pack_forget()
        else:
            self.multi_gpu_frame.pack(fill="x", padx=10, pady=5)
            self.main_gpu_dropdown.configure(state="normal")
        self.auto_calc_ts()

    def on_spec_type_changed(self, choice):
        if choice == "none":
            self.spec_sub_frame.pack_forget()
            self.draft_model_row.pack_forget()
        else:
            self.spec_sub_frame.pack(side="left")
            if choice == "draft-mtp":
                self.draft_model_row.pack_forget()
            else:
                try:
                    self.draft_model_row.pack(fill="x", padx=10, pady=2, before=self.draft_model_row.master.winfo_children()[-1])
                except:
                    self.draft_model_row.pack(fill="x", padx=10, pady=2)
        if choice == "dflash":
            try:
                self.dflash_frame.pack(fill="x", padx=10, pady=2, before=self.extra_args_frame)
            except:
                pass
            self.toggle_dflash_params()
        else:
            self.dflash_frame.pack_forget()

    def toggle_moe_frame(self):
        if self.is_moe.get() == "on":
            try:
                self.moe_frame.pack(fill="x", padx=10, pady=2, before=self.extra_args_frame)
            except:
                pass
        else:
            self.moe_frame.pack_forget()

    def toggle_dflash_params(self):
        if self.spec_dflash_default.get() == "on":
            self.dflash_params_frame.pack_forget()
        else:
            self.dflash_params_frame.pack(side="left")

    def auto_calc_ts(self, _=None):
        sel = self.gpu_selection.get()
        try: main_v = int(self.ts_main_val.get())
        except: main_v = 28
        if "所有显卡" in sel:
            m_idx = int(self.main_gpu_index.get()); ts_list = [0] * len(self.available_gpus)
            other_v = max(0, 36 - main_v)
            for i in range(len(ts_list)): ts_list[i] = main_v if i == m_idx else other_v // (len(ts_list)-1 if len(ts_list)>1 else 1)
            self.ts_final_str.set(",".join(map(str, ts_list)))
        else: self.ts_final_str.set("1")
        self.ts_display.configure(text=f"-> TS: {self.ts_final_str.get()}")

    def create_input(self, parent, label, var, browse=False):
        self.create_dir_input(parent, label, var, browse_cmd=lambda: self.browse(var), extra_btn=None, extra_cmd=None)

    def create_dir_input(self, parent, label, var, browse_cmd=None, extra_btn=None, extra_cmd=None):
        self.create_dir_input_row(parent, label, var, browse_cmd, extra_btn, extra_cmd)

    def create_dir_input_row(self, parent, label, var, browse_cmd=None, extra_btn=None, extra_cmd=None):
        f = ctk.CTkFrame(parent, fg_color="transparent"); f.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(f, text=label, width=90, anchor="w").pack(side="left", padx=(5, 2))
        ctk.CTkEntry(f, textvariable=var).pack(side="left", fill="x", expand=True, padx=(5, 5))
        if browse_cmd:
            ctk.CTkButton(f, text="...", width=40, command=browse_cmd).pack(side="right", padx=(0, 5))
        if extra_btn and extra_cmd:
            ctk.CTkButton(f, text=extra_btn, width=45, command=extra_cmd).pack(side="right")
        return f

    def create_small_input(self, parent, label, var, width=70):
        ctk.CTkLabel(parent, text=label).pack(side="left", padx=(5, 2))
        ctk.CTkEntry(parent, textvariable=var, width=width).pack(side="left", padx=5)

    def create_small_option(self, parent, label, var, values, width=70, command=None):
        ctk.CTkLabel(parent, text=label).pack(side="left", padx=(5, 2))
        dropdown = ctk.CTkOptionMenu(parent, variable=var, values=values, width=width, command=command)
        dropdown.pack(side="left", padx=5)
        return dropdown

    def create_model_select_row(self, parent, label, var, values, extra_label=None, extra_var=None, extra_values=None, extra_width=150):
        f = ctk.CTkFrame(parent, fg_color="transparent"); f.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(f, text=label, width=90, anchor="w").pack(side="left")
        dropdown = ctk.CTkOptionMenu(f, values=values, variable=var, width=0)
        dropdown.pack(side="left", fill="x", expand=True, padx=(5, 5))
        extra_dropdown = None
        if extra_label and extra_var is not None:
            extra_dropdown = self.create_small_option(f, extra_label, extra_var, extra_values, width=extra_width)
        return f, dropdown, extra_dropdown

    def create_small_check(self, parent, label, var, text="启用", width=10):
        ctk.CTkLabel(parent, text=label).pack(side="left", padx=(5, 2))
        ctk.CTkCheckBox(parent, text=text, variable=var, onvalue="on", offvalue="off", width=width).pack(side="left", padx=5)

    def browse(self, var):
        p = filedialog.askopenfilename()
        if p: var.set(p)

if __name__ == "__main__":
    try:
        app = LlamaLauncherV6()
        app.mainloop()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] 程序崩溃！\n{traceback.format_exc()}")
        input("按回车键关闭控制台...")
