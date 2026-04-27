import os
import subprocess
import threading
import time
import queue
import yaml
import platform
import re
import customtkinter as ctk
from tkinter import filedialog
import psutil
import cpuinfo

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except:
    HAS_NVML = False

import sys

BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.yml")

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

    def get_system_gpus(self):
        gpus = []
        if HAS_NVML:
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                # 兼容处理显卡名称字节转字符串
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes): name = name.decode('utf-8')
                gpus.append({"index": i, "name": name})
        return gpus

    def setup_ui(self):
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # 配置选择栏 (左上角)
        cfg_bar = ctk.CTkFrame(self.main_container, height=35, fg_color="#1a1a1a")
        cfg_bar.pack(fill="x", pady=(0, 8))
        cfg_bar.pack_propagate(False)

        ctk.CTkLabel(cfg_bar, text="配置:", width=50, anchor="w").pack(side="left", padx=(5, 2))

        profile_values = self.config_profiles if self.config_profiles else ["(无)"]
        self.profile_dropdown = ctk.CTkOptionMenu(cfg_bar, variable=self.current_profile, values=profile_values, width=180, command=self.on_profile_selected)
        self.profile_dropdown.pack(side="left", padx=(0, 5))

        ctk.CTkEntry(cfg_bar, textvariable=self.new_config_name, placeholder_text="新配置名称").pack(side="left", fill="x", expand=True, padx=2)
        ctk.CTkButton(cfg_bar, text="+新增", width=45, command=self.add_new_profile).pack(side="right", padx=(0, 2))
        ctk.CTkButton(cfg_bar, text="保存", width=45, command=self.save_config).pack(side="right")

        # --- 需求 3: 将监控部分放在程序路径上面 ---
        self.monitor_frame = ctk.CTkFrame(self.main_container, height=120, fg_color="#1a1a1a")
        self.monitor_frame.pack(fill="x", pady=(0, 15))
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
        path_frame.pack(fill="x", pady=5)
        self.create_input(path_frame, "程序路径:", self.server_path, browse=True)

        # Main model directory + selection
        self.create_dir_input(path_frame, "模型目录:", self.model_dir,
                              browse_cmd=lambda: self.browse_dir(self.model_dir),
                              extra_btn="刷新", extra_cmd=self.refresh_models)

        # Main model + mmproj selection row
        self.model_sel_row, self.model_dropdown, self.mmproj_dropdown = self.create_model_select_row(
            path_frame, "模型选择:", self.model_name, [],
            extra_label="多模态:", extra_var=self.mmproj_name, extra_values=["(无)"], extra_width=150)

        # Draft model directory + selection
        self.draft_dir_row = self.create_dir_input_row(path_frame, "投机目录:", self.draft_model_dir,
                                                       browse_cmd=lambda: self.browse_dir(self.draft_model_dir),
                                                       extra_btn="刷新", extra_cmd=self.refresh_draft_models)

        self.draft_sel_row, self.draft_model_dropdown, _ = self.create_model_select_row(
            path_frame, "投机模型:", self.draft_model_name, ["(无)"])

        self.update_draft_visibility()

        param_tabs = ctk.CTkTabview(self.main_container)
        param_tabs.pack(fill="x", pady=10)

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
        self.main_gpu_dropdown = self.create_small_option(row_gpu, "主卡编号:", self.main_gpu_index,
                                                          [str(g['index']) for g in self.available_gpus])
        self.create_small_input(row_gpu, "GPU卸载层数:", self.ngl)
        self.create_small_option(row_gpu, "拆分模式:", self.split_mode, ["layer", "none", "row"], width=60)
        self.create_small_input(row_gpu, "主卡配比:", self.ts_main_val)
        self.ts_display = ctk.CTkLabel(row_gpu, text="-> TS: --", text_color="#3498db")
        self.ts_display.pack(side="left", padx=10)
        self.create_small_option(row_gpu, "Flash Attention:", self.flash_attn, ["auto", "on", "off"], width=60)

        # Tab 3: 采样/生成
        tab_sample = param_tabs.add("采样/生成")
        row_sample = ctk.CTkFrame(tab_sample, fg_color="transparent")
        row_sample.pack(fill="x", padx=10, pady=5)
        self.create_small_input(row_sample, "温度:", self.temperature)
        self.create_small_input(row_sample, "限制累计概率:", self.top_p)
        self.create_small_input(row_sample, "限制候选数量:", self.top_k)
        self.create_small_input(row_sample, "重复惩罚:", self.repeat_penalty)
        self.create_small_input(row_sample, "种子:", self.seed)
        self.create_small_input(row_sample, "预测Token:", self.n_predict)
        self.create_small_option(row_sample, "Mirostat算法:", self.mirostat, ["0", "1", "2"], width=60)

        # Tab 4: 高级
        tab_adv = param_tabs.add("高级")
        row_adv = ctk.CTkFrame(tab_adv, fg_color="transparent")
        row_adv.pack(fill="x", padx=10, pady=5)
        self.create_small_input(row_adv, "并发数:", self.np_val)
        self.create_small_input(row_adv, "DFlash:", self.draft_max)
        self.create_small_input(row_adv, "卸载层数:", self.ngld, width=70)
        self.create_small_input(row_adv, "线程数:", self.threads)
        self.create_small_input(row_adv, "Prompt大小:", self.batch_size)
        self.create_small_input(row_adv, "UBatch大小:", self.ubatch_size)
        self.create_small_option(row_adv, "K量化：", self.kv_quant_k, self.cache_type_options)
        self.create_small_option(row_adv, "V量化：", self.kv_quant_v, self.cache_type_options)

        row_adv2 = ctk.CTkFrame(tab_adv, fg_color="transparent")
        row_adv2.pack(fill="x", padx=10, pady=5)
        self.create_small_check(row_adv2, "内存映射模型:", self.mmap)
        self.create_small_check(row_adv2, "性能计时：", self.perf_timer, text="开启")

        # 额外参数输入框（独占一行，填满宽度）
        f = ctk.CTkFrame(tab_adv, fg_color="transparent")
        f.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(f, text="额外参数:", width=90, anchor="w").pack(side="left")
        ctk.CTkEntry(f, textvariable=self.extra_args).pack(side="left", fill="x", expand=True, padx=(5, 5))

        # 命令预览 和 日志区
        log_box = ctk.CTkFrame(self.main_container, fg_color="transparent")
        log_box.pack(fill="both", expand=True, pady=10)

        # 命令预览
        ctk.CTkLabel(log_box, text="[ 命令行预览 ]", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.cmd_display = ctk.CTkTextbox(
                log_box,
                font=("Consolas", 10),  # 稍微缩小预览字体，更显精炼
                text_color="#3498db",
                height=70,              # 固定高度，防止它无限拉伸
                activate_scrollbars=False # 预览较短时可关闭滚动条，视觉更统一
            )
        # 缩小上方间距
        self.cmd_display.pack(fill="x", pady=(2, 8))
        # 初始设为只读
        self.cmd_display.configure(state="disabled")

        # 日志区
        ctk.CTkLabel(log_box, text="[ 实时运行日志 ]", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.log_display = ctk.CTkTextbox(log_box, font=("Consolas", 11), fg_color="#0d0d0d")
        self.log_display.pack(fill="both", expand=True, pady=5)

        # 配置日志颜色标签
        self.log_display.tag_config("info", foreground="#2ecc71")
        self.log_display.tag_config("warn", foreground="#f1c40f")
        self.log_display.tag_config("error", foreground="#e74c3c")
        self.log_display.tag_config("highlight", foreground="#3498db")

        # 4. 按钮
        btn_row = ctk.CTkFrame(self.main_container, fg_color="transparent")
        btn_row.pack(fill="x", pady=5)
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
          self.mmproj_name, self.draft_model_dir, self.draft_model_name,
            self.host, self.port, self.ngl, self.ctx_custom,
            self.ts_final_str, self.kv_quant_k, self.kv_quant_v, self.reasoning,
            self.gpu_selection, self.main_gpu_index ,self.np_val,
          self.draft_max, self.perf_timer, self.mmap, self.flash_attn, self.split_mode,
            self.threads, self.batch_size, self.ubatch_size, self.ngld,
             self.temperature, self.top_p, self.top_k, self.repeat_penalty,
             self.seed, self.n_predict, self.mirostat, self.extra_args
        ]
        for var in vars_to_track:
            var.trace_add("write", lambda *args: self.update_cmd_preview())
        self.draft_max.trace_add("write", self.on_draft_max_changed)

    def on_draft_max_changed(self, *_):
        self.update_draft_visibility()

    def update_draft_visibility(self):
        show = self.draft_max.get().strip() != "0"
        if show:
            self.draft_dir_row.pack(fill="x", padx=10, pady=2)
            self.draft_sel_row.pack(fill="x", padx=10, pady=2)
        else:
            self.draft_dir_row.pack_forget()
            self.draft_sel_row.pack_forget()

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

    def refresh_draft_models(self):
        d = self.draft_model_dir.get().strip()
        if not d or not os.path.isdir(d):
            return
        all_files = sorted(os.listdir(d))
        model_files = [f for f in all_files if not f.startswith("mmproj") and not f.startswith(".")]

        current_draft = self.draft_model_name.get()
        model_opts = model_files if model_files else ["(无)"]
        self.draft_model_dropdown.configure(values=model_opts)
        if current_draft not in model_opts and model_files:
            self.draft_model_name.set(model_files[0])
        elif current_draft in model_opts:
            self.draft_model_name.set(current_draft)

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

    def get_full_draft_path(self):
        d = self.draft_model_dir.get().strip()
        n = self.draft_model_name.get().strip()
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
        """构建完整的命令，包括环境变量前缀和llama-server参数"""
        # 1. 构建环境变量前缀
        env_prefix = ""
        if platform.system() == "Windows":
            env_prefix = f"set CUDA_VISIBLE_DEVICES={self.get_cuda_visible_devices()} && "
        else:
            env_prefix = f"export CUDA_VISIBLE_DEVICES={self.get_cuda_visible_devices()} && "

        # 2. 构建llama-server命令
        mg_idx = self.main_gpu_index.get() if self.main_gpu_index.get() else "0"
        quote = lambda s: f'"{s}"' if for_display else s

        cmd = [
            quote(self.server_path.get()),
            "--host", self.host.get(),
            "--port", self.port.get(),
            "-m", quote(self.get_full_model_path()),
            "-ngl", self.ngl.get(),
            "-mg", mg_idx,
            "-c", self.ctx_custom.get(),
            "-ts", self.ts_final_str.get(),
            "-sm", self.split_mode.get(),
            "-np", self.np_val.get(),
            "-ctk", self.kv_quant_k.get(),
            "-ctv", self.kv_quant_v.get(),
            "-t", self.threads.get(),
            "-b", self.batch_size.get(),
            "-ub", self.ubatch_size.get(),
            "--temp", self.temperature.get(),
            "--top-p", self.top_p.get(),
            "--top-k", self.top_k.get(),
            "--repeat-penalty", self.repeat_penalty.get(),
            "-s", self.seed.get(),
            "-n", self.n_predict.get(),
        ]
        if int(self.mirostat.get()) != 0:
            cmd.extend(["--mirostat", self.mirostat.get()])
        if self.reasoning.get() == "on":
            cmd.extend(["--reasoning", "on"])
        else:
            cmd.extend(["--reasoning", "off"])

        draft_val = self.draft_max.get().strip()
        if draft_val and draft_val != "0":
            cmd.extend(["--draft", draft_val])

        if self.perf_timer.get() == "on":
            cmd.append("--perf")

        if self.mmap.get() != "on":
            cmd.append("--no-mmap")
        else:
            cmd.append("--mmap")

        if self.flash_attn.get() != "auto":
            cmd.extend(["-fa", self.flash_attn.get()])

        mmproj_full = self.get_full_mmproj_path()
        if mmproj_full:
            cmd.extend(["-mm", quote(mmproj_full)])
        draft_full = self.get_full_draft_path()
        if draft_val and draft_val != "0" and draft_full:
            cmd.extend(["-md", quote(draft_full)])

        # ngld参数，当DFlash不为0时生效（auto/0/1/2/4等）
        if draft_val and draft_val != "0":
            ngld_val = self.ngld.get().strip()
            cmd.extend(["--ngld", ngld_val])

        # 额外参数，按空格分割追加到命令末尾（不添加引号）
        extra = self.extra_args.get().strip()
        if extra:
            for arg in re.split(r'\s+', extra):
                cmd.append(arg)

        # 3. 返回完整命令字符串（含环境变量）或纯命令列表
        if for_display:
            return env_prefix + " ".join(cmd)
        return cmd

    def update_cmd_preview(self):
        """更新命令行预览显示"""
        cmd_str = self.build_command_list(for_display=True)
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
                            h = pynvml.nvmlDeviceGetHandleByIndex(i)
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

        # 需求 4: 解析 llama.cpp 的 Token 输出
        # 典型输出: print_statistics: ... tokens, 5.23 t/s
        if "tokens" in text and "t/s" in text:
            try:
                t_match = re.search(r"(\d+) tokens", text)
                s_match = re.search(r"([\d\.]+) t/s", text)
                if t_match: self.total_tokens += int(t_match.group(1))
                if s_match: self.tokens_per_sec = s_match.group(1)

                # 计算上下文占比
                ctx_limit = int(self.ctx_custom.get()) if self.ctx_custom.get().isdigit() else 32768
                percent = min(100, int((self.total_tokens / ctx_limit) * 100))

                print(f"[Token] Tokens: {self.total_tokens} | 速度: {self.tokens_per_sec} t/s | 占比: {percent}%")
            except: pass

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
        self.total_tokens = 0 # 重置计数
        self.log_display.delete("1.0", "end")

        # 1. 准备环境变量
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.get_cuda_visible_devices()

        # 2. 构建命令
        cmd = self.build_command_list(for_display=False)

        def run():
            self.process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
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
            "draft_model_dir": "", "draft_model_name": "",
            "host": "0.0.0.0", "port": "8080",
            "ngl": "all", "ctx": "32768", "ts_ratio": "28",
            "cache_type": "q8_0", "np_val": "1",
            "mmap": "off", "perf_timer": "off", "draft_max": "0"
        }

        self.server_path = ctk.StringVar(value=self.defaults["server_path"])
        self.model_dir = ctk.StringVar()
        self.model_name = ctk.StringVar()
        self.mmproj_name = ctk.StringVar()
        self.draft_model_dir = ctk.StringVar()
        self.draft_model_name = ctk.StringVar()
        self.host = ctk.StringVar(value=self.defaults["host"])
        self.port = ctk.StringVar(value=self.defaults["port"])
        self.ngl = ctk.StringVar(value=self.defaults["ngl"])
        self.ctx_custom = ctk.StringVar(value=self.defaults["ctx"])
        self.ts_main_val = ctk.StringVar(value=self.defaults["ts_ratio"])
        self.kv_quant_k = ctk.StringVar(value=self.defaults["cache_type"])
        self.kv_quant_v = ctk.StringVar(value=self.defaults["cache_type"])
        self.np_val = ctk.StringVar(value=self.defaults["np_val"])
        self.flash_attn = ctk.StringVar(value="auto")
        self.split_mode = ctk.StringVar(value="layer")
        self.draft_max = ctk.StringVar(value="0")
        self.perf_timer = ctk.StringVar(value="off")
        self.mmap = ctk.StringVar(value="on")
        self.threads = ctk.StringVar(value="-1")
        self.batch_size = ctk.StringVar(value="2048")
        self.ubatch_size = ctk.StringVar(value="512")
        self.temperature = ctk.StringVar(value="0.8")
        self.top_p = ctk.StringVar(value="0.95")
        self.top_k = ctk.StringVar(value="40")
        self.repeat_penalty = ctk.StringVar(value="1.0")
        self.seed = ctk.StringVar(value="-1")
        self.n_predict = ctk.StringVar(value="-1")
        self.mirostat = ctk.StringVar(value="0")
        self.ctx_preset = ctk.StringVar(value="自定义")
        self.reasoning = ctk.StringVar(value="off")
        self.cache_type_options = ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
        self.ngld = ctk.StringVar(value="auto")
        self.extra_args = ctk.StringVar()
        self.ts_final_str = ctk.StringVar(value="1")

        gpu_opts = [f"{g['index']}: {g['name']}" for g in self.available_gpus]
        if len(gpu_opts) > 1: gpu_opts.append("所有显卡 (并行)")
        self.gpu_selection = ctk.StringVar(value=gpu_opts[-1])
        self.main_gpu_index = ctk.StringVar(value="0")

        self.var_map = {
            "server_path": self.server_path, "model_dir": self.model_dir,
            "model_name": self.model_name, "mmproj_name": self.mmproj_name,
            "draft_model_dir": self.draft_model_dir, "draft_model_name": self.draft_model_name,
            "host": self.host, "port": self.port, "ngl": self.ngl,
            "ctx": self.ctx_custom, "ts_ratio": self.ts_main_val,
            "cache_type_k": self.kv_quant_k, "cache_type_v": self.kv_quant_v,
            "np_val": self.np_val, "mmap": self.mmap, "draft_max": self.draft_max,
            "perf_timer": self.perf_timer, "flash_attn": self.flash_attn,
            "split_mode": self.split_mode, "threads": self.threads,
            "batch_size": self.batch_size, "ubatch_size": self.ubatch_size,
            "temperature": self.temperature, "top_p": self.top_p,
            "top_k": self.top_k, "repeat_penalty": self.repeat_penalty,
            "seed": self.seed, "n_predict": self.n_predict,
            "mirostat": self.mirostat, "reasoning": self.reasoning,
            "ngld": self.ngld, "extra_args": self.extra_args,
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
                elif str_val:
                    var.set(str_val)
        cache_opts = cfg.get("cache_type_options")
        if isinstance(cache_opts, list):
            self.cache_type_options = cache_opts

    def on_profile_selected(self, _=None):
        name = self.current_profile.get()
        if not name or name == "(无)": return
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            all_cfg = yaml.safe_load(f) or {}
        profiles = {k: v for k, v in all_cfg.items() if isinstance(v, dict)}
        if name in profiles:
            self.apply_profile(name, profiles[name])

    def add_new_profile(self):
        name = self.new_config_name.get().strip()
        if not name: return
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
            idx = choice.split(":")[0]; self.main_gpu_index.set(idx); self.main_gpu_dropdown.configure(state="disabled")
        else: self.main_gpu_dropdown.configure(state="normal")
        self.auto_calc_ts()

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
        ctk.CTkLabel(f, text=label, width=90, anchor="w").pack(side="left")
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
    app = LlamaLauncherV6()
    app.mainloop()
