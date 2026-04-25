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

CONFIG_FILE = "config.yml"

class LlamaLauncherV6(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Llama-CPP 启动器 V1.0")
        self.geometry("1300x900")
        ctk.set_appearance_mode("dark")
        
        self.available_gpus = self.get_system_gpus()
        
        # 核心修复：先初始化变量（load_config），再创建 UI
        self.load_config() 
        
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
        self.create_input(path_frame, "模型路径:", self.model_path, browse=True)
        self.create_input(path_frame, "多模态路径:", self.mmproj_path, browse=True)

        param_grid = ctk.CTkFrame(self.main_container)
        param_grid.pack(fill="x", pady=10)

        row1 = ctk.CTkFrame(param_grid, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=5)
        self.create_small_input(row1, "地址:", self.host)
        self.create_small_input(row1, "端口:", self.port)
        self.create_small_input(row1, "NGL:", self.ngl)
        ctk.CTkLabel(row1, text="上下文:").pack(side="left", padx=(20, 2))
        ctk.CTkOptionMenu(row1, variable=self.ctx_preset, values=["8192", "32768", "65536", "131072", "262144", "自定义"],
                         command=lambda c: self.ctx_custom.set(c) if c!="自定义" else None, width=100).pack(side="left", padx=5)
        ctk.CTkEntry(row1, textvariable=self.ctx_custom, width=80).pack(side="left")

        row2 = ctk.CTkFrame(param_grid, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(row2, text="运行设备:").pack(side="left", padx=(5, 2))
        gpu_opts = [f"{g['index']}: {g['name']}" for g in self.available_gpus]
        if len(self.available_gpus) > 1: gpu_opts.append("所有显卡 (并行)")
        self.gpu_dropdown = ctk.CTkOptionMenu(row2, variable=self.gpu_selection, values=gpu_opts, command=self.sync_main_gpu)
        self.gpu_dropdown.pack(side="left", padx=5)

        ctk.CTkLabel(row2, text="主卡编号:").pack(side="left", padx=(15, 2))
        self.main_gpu_dropdown = ctk.CTkOptionMenu(row2, variable=self.main_gpu_index, 
                                                 values=[str(g['index']) for g in self.available_gpus], width=70)
        self.main_gpu_dropdown.pack(side="left", padx=5)

        ctk.CTkLabel(row2, text="主卡配比:").pack(side="left", padx=(15, 2))
        ctk.CTkEntry(row2, textvariable=self.ts_main_val, width=50).pack(side="left", padx=5)
        self.ts_display = ctk.CTkLabel(row2, text="-> TS: --", text_color="#3498db")
        self.ts_display.pack(side="left", padx=10)

        ctk.CTkLabel(row2,  text="KV量化：").pack(side="left", padx=(15, 2))
        ctk.CTkOptionMenu(row2, variable=self.kv_quant, values=self.cache_type_options, width=90).pack(side="left", padx=5)

        ctk.CTkLabel(row2,  text="思考模式：").pack(side="left", padx=(15, 2))
        ctk.CTkCheckBox(row2, text="开启", variable=self.reasoning, width=90).pack(side="left", padx=10)
    
        # 新增优化选项行
        row3 = ctk.CTkFrame(param_grid, fg_color="transparent")
        row3.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(row3, text="并发数:").pack(side="left", padx=(5, 2))
        ctk.CTkEntry(row3, textvariable=self.np_val, width=50).pack(side="left", padx=5)

        ctk.CTkLabel(row3,  text="优化 Flash Attention:").pack(side="left", padx=(5, 2))
        ctk.CTkCheckBox(row3, text="开启", variable=self.flash_attn, width=90).pack(side="left", padx=10)

        ctk.CTkLabel(row3,  text="性能计时：").pack(side="left", padx=(5, 2))
        ctk.CTkCheckBox(row3, text="开启", variable=self.perf_timer, width=90).pack(side="left", padx=10)

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
        self.start_btn = ctk.CTkButton(btn_row, text="🚀 启动服务并保存配置", fg_color="#2ecc71", command=self.start_server, height=40)
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
            self.server_path, self.model_path, self.mmproj_path,
            self.host, self.port, self.ngl, self.ctx_custom,
            self.ts_final_str, self.kv_quant, self.reasoning,
            self.gpu_selection, self.main_gpu_index ,self.np_val,
            self.flash_attn, self.perf_timer
        ]
        for var in vars_to_track:
            var.trace_add("write", lambda *args: self.update_cmd_preview())

    def update_cmd_preview(self):
        # 1. 处理环境变量显示逻辑
        sel = self.gpu_selection.get()
        if "所有显卡" in sel:
            visible_ids = ",".join([str(g['index']) for g in self.available_gpus])
        else:
            visible_ids = sel.split(":")[0]
        
        # 根据系统平台显示不同的环境变量设置命令
        env_prefix = f"set CUDA_VISIBLE_DEVICES={visible_ids} && " if platform.system() == "Windows" else f"export CUDA_VISIBLE_DEVICES={visible_ids} && "

        # 获取当前主卡索引
        mg_idx = self.main_gpu_index.get() if self.main_gpu_index.get() else "0"
        # 构建命令列表
        cmd = [
            f'"{self.server_path.get()}"',
            "--host", self.host.get(),
            "--port", self.port.get(),
            "-m", f'"{self.model_path.get()}"',
            "-ngl", self.ngl.get(),
            "-mg", mg_idx,
            "-c", self.ctx_custom.get(),
            "-ts", self.ts_final_str.get(),
            "-np", self.np_val.get(),
            "--cache-type-k", self.kv_quant.get(),
            "--cache-type-v", self.kv_quant.get(),
        ]
        if self.reasoning.get() == "1" or self.reasoning.get() == "on":
            cmd.extend(["--reasoning", "on"])
        else:
            cmd.extend(["--reasoning", "off"])

        if self.flash_attn.get() == "1" or self.flash_attn.get() == "on":
            cmd.append("--flash-attn")
            cmd.append("on")

        if self.perf_timer.get() == "1" or self.perf_timer.get() == "on":
            cmd.append("--perf")

        if self.mmproj_path.get():
            cmd.extend(["-mm", f'"{self.mmproj_path.get()}"'])
        
        # 4. 组合最终显示的字符串
        full_display_str = env_prefix + " ".join(cmd)
        
        # 更新文本框内容
        self.cmd_display.configure(state="normal")
        self.cmd_display.delete("1.0", "end")
        self.cmd_display.insert("1.0", full_display_str)
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
                
                self.token_info.configure(text=f"Tokens: {self.total_tokens} | 速度: {self.tokens_per_sec} t/s | 占比: {percent}%")
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
        self.save_config()
        self.log_display.delete("1.0", "end")

        # 1. 准备环境变量
        env = os.environ.copy()
        sel = self.gpu_selection.get()

        if "所有显卡" in sel:
            # 暴露所有可用显卡的 ID
            visible_ids = [str(g['index']) for g in self.available_gpus]
            env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_ids)
        else:
            # 仅暴露选中的单张显卡 ID
            target_id = sel.split(":")[0]
            env["CUDA_VISIBLE_DEVICES"] = target_id

        # 2. 构建命令
        cmd = [
            f'"{self.server_path.get()}"',
            "--host", self.host.get(),
            "--port", self.port.get(),
            "-m", f'"{self.model_path.get()}"',
            "-ngl", self.ngl.get(),
            "-mg", self.main_gpu_index.get(), # 指定主卡
            "-c", self.ctx_custom.get(),
            "-ts", self.ts_final_str.get(),
            "-np", self.np_val.get(),
            "--cache-type-k", self.kv_quant.get(),
            "--cache-type-v", self.kv_quant.get(),
        ]
        if self.reasoning.get() == "1" or self.reasoning.get() == "on":
            cmd.extend(["--reasoning", "on"])
        else:
            cmd.extend(["--reasoning", "off"])
        if self.mmproj_path.get(): cmd.extend(["-mm", f'"{self.mmproj_path.get()}"'])

        if self.flash_attn.get() == "1" or self.flash_attn.get() == "on":
            cmd.extend(["--flash-attn", "on"])
        elif self.flash_attn.get() == "0" or self.flash_attn.get() == "off":
            cmd.extend(["--flash-attn", "off"])
        if self.perf_timer.get() == "1" or self.perf_timer.get() == "on":
            cmd.append("--perf")

        env = os.environ.copy()
        sel = self.gpu_selection.get()
        env["CUDA_VISIBLE_DEVICES"] = ",".join([str(g['index']) for g in self.available_gpus]) if "所有" in sel else sel.split(":")[0]

        def run():
            self.process = subprocess.Popen(" ".join(cmd), env=env, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            self.running = True
            for line in iter(self.process.stdout.readline, ""):
                self.log_queue.put(line)
            self.running = False

        threading.Thread(target=run, daemon=True).start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="enabled")

    def stop_server(self):
        if self.process:
            # 执行强制结束命令
            os.system("taskkill /IM llama-server.exe /F")
        
            # 发送停止成功的提示到日志队列
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

    def load_config(self):
        default_config = {
            "server_path": r"D:\Program Files\llama\llama-server.exe",
            "model_path": "",
            "mmproj_path": "",
            "host": "0.0.0.0",
            "port": "8080",
            "ngl": "all",
            "ctx": "32768",
            "ts_ratio": "28",
            "cache_type": "q8_0",
            "np_val": "1"
        }

        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                user_cfg = yaml.safe_load(f)
                if user_cfg: default_config.update(user_cfg)
        
        self.server_path = ctk.StringVar(value=default_config["server_path"])
        self.model_path = ctk.StringVar(value=default_config["model_path"])
        self.mmproj_path = ctk.StringVar(value=default_config["mmproj_path"])
        self.host = ctk.StringVar(value=default_config["host"])
        self.port = ctk.StringVar(value=default_config["port"])
        self.ngl = ctk.StringVar(value=default_config["ngl"])
        self.ctx_custom = ctk.StringVar(value=default_config["ctx"])
        self.ts_main_val = ctk.StringVar(value=default_config["ts_ratio"])
        self.kv_quant = ctk.StringVar(value=default_config["cache_type"])
        self.np_val = ctk.StringVar(value=default_config["np_val"])
        self.flash_attn = ctk.StringVar(value="off")
        self.perf_timer = ctk.StringVar(value="off")
        self.ctx_preset = ctk.StringVar(value="自定义")
        self.reasoning = ctk.StringVar(value="off")
        self.cache_type_options = default_config.get("cache_type_options", ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"])
        self.ts_final_str = ctk.StringVar(value="1")

        gpu_opts = [f"{g['index']}: {g['name']}" for g in self.available_gpus]
        if len(gpu_opts) > 1: gpu_opts.append("所有显卡 (并行)")
        self.gpu_selection = ctk.StringVar(value=gpu_opts[-1])
        self.main_gpu_index = ctk.StringVar(value="0")

    def save_config(self):
        """保存所有当前参数到 YAML，保留原文件中未被覆盖的字段"""
        cfg = {}
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}

        # 更新 GUI 管理的字段
        cfg.update({
            "server_path": self.server_path.get(),
            "model_path": self.model_path.get(),
            "mmproj_path": self.mmproj_path.get(),
            "host": self.host.get(),
            "port": self.port.get(),
            "ngl": self.ngl.get(),
            "ctx": self.ctx_custom.get(),
            "ts_ratio": self.ts_main_val.get(),
            "np_val": self.np_val.get(),
            "cache_type": self.kv_quant.get(),
        })

        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

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
        f = ctk.CTkFrame(parent, fg_color="transparent"); f.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(f, text=label, width=90, anchor="w").pack(side="left")
        ctk.CTkEntry(f, textvariable=var).pack(side="left", fill="x", expand=True, padx=10)
        if browse: ctk.CTkButton(f, text="...", width=40, command=lambda: self.browse(var)).pack(side="right")

    def create_small_input(self, parent, label, var):
        ctk.CTkLabel(parent, text=label).pack(side="left", padx=(5, 2))
        ctk.CTkEntry(parent, textvariable=var, width=70).pack(side="left", padx=5)

    def browse(self, var):
        p = filedialog.askopenfilename()
        if p: var.set(p)

if __name__ == "__main__":
    app = LlamaLauncherV6()
    app.mainloop()