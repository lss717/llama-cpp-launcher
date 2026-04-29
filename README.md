# Llama-CPP Server Launcher

图形化的 llama.cpp server 启动工具，支持多配置管理、投机解码、实时资源监控和全参数可视化配置。

## 功能概览

### 多配置 Profile 管理
- 支持多个预设配置文件（Profile），通过下拉菜单切换
- 一键新增/保存配置，所有参数持久化到 `config.yml`
- 启动时自动加载上次使用的 Profile

### 系统实时监控
- **CPU**：处理器型号 + 实时占用率
- **内存**：已用 / 总量 / 百分比
- **GPU**（多卡）：核心利用率 | 显存占用 | 温度 | 功耗

### 智能显卡管理
- 自动检测 NVIDIA 显卡，支持单卡或多卡并行
- 自动计算 Tensor Split (`-ts`)，可设置主卡权重配比
- 自动配置 `CUDA_VISIBLE_DEVICES` 环境变量

### 模型目录浏览
- **主模型**：设置目录后一键刷新，自动列出所有 GGUF 文件
- **多模态投影器 (mmproj)**：自动识别 `mmproj-*` 文件
- **投机草稿模型**：独立目录管理，支持 DFlash 等加速方案

### 分 Tab 参数配置
| Tab | 包含参数 |
|-----|---------|
| **API/基础** | 地址、端口、上下文长度（预设+自定义）、思考模式 |
| **GPU/加速** | 运行设备、主卡编号、GPU层数(-ngl)、拆分模式(-sm)、Flash Attention、TS配比 |
| **采样/生成** | 温度、top-p、top-k、重复惩罚、随机种子、预测Token数、Mirostat |
| **高级** | 并发数、投机解码(DFlash)、草稿模型GPU层数、线程数、批处理大小、KV量化类型(K/V独立)、内存映射、性能计时、额外参数 |

### KV 缓存量化支持
f32 / f16 / bf16 / q8_0 / q4_0 / q4_1 / iq4_nl / q5_0 / q5_1 / turbo2 / turbo3 / turbo4（K/V 独立设置）

### 投机解码 (Speculative Decoding)
- 设置 `--draft` 参数启用草稿 Token 数量
- 支持配置独立的草稿模型目录和文件选择
- 可调节草稿模型的 GPU 卸载层数 (`--ngld`)

### 实时日志与统计
- Token 数量、生成速率 (tokens/s)、上下文占用百分比
- 彩色高亮：错误(红)、警告(黄)、llama.cpp 输出(蓝)

## 安装使用

### 环境要求
- Python 3.8+
- NVIDIA 显卡（可选，用于 GPU 监控）

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行源码模式
```bash
python llama_launcher_gui.py
```

## 配置说明 (`config.yml`)

配置文件按 Profile 组织，每个 Profile 包含一组完整参数：

```yaml
MyModel:
  server_path: "D:/Program Files/llama/llama-server.exe"
  model_dir: "F:/models/my-model"
  model_name: "model-q4_k_m.gguf"
  mmproj_name: "(无)"
  draft_model_dir: ""
  draft_model_name: "(无)"
  host: "0.0.0.0"
  port: "8080"
  ngl: "all"
  ctx: "32768"
  ts_ratio: "28"
  cache_type_k: "q8_0"
  cache_type_v: "q8_0"
  np_val: "-1"
  mmap: "on"
  draft_max: "0"
  perf_timer: "off"
  flash_attn: "auto"
  split_mode: "layer"
  threads: "-1"
  batch_size: "2048"
  ubatch_size: "512"
  temperature: "0.8"
  top_p: "0.95"
  top_k: "40"
  repeat_penalty: "1.0"
  seed: "-1"
  n_predict: "-1"
  mirostat: "0"
  reasoning: "off"
```

## 使用流程

1. **选择 Profile**：从顶部下拉菜单选择已有配置，或新建一个
2. **设置路径**：指定 `llama-server.exe`、模型目录（点击刷新加载模型列表）
3. **选择设备**：单卡直接选，多卡可设主卡和权重配比
4. **调整参数**：按 Tab 页按需修改，命令行预览实时更新
5. **启动/停止**：点击按钮控制服务，日志区实时输出运行状态

## 打包为 EXE

将项目打包为独立可执行文件，无需 Python 环境即可运行。`config.yml` **不嵌入** exe，而是放在 exe 同级目录下外部读取和编辑。

### 构建步骤
1. **安装所有依赖（含 PyInstaller）**：
   ```bash
   pip install -r requirements.txt
   ```
2. **进入项目目录**：`cd llama-cpp-launcher`
3. **执行打包命令**：
   ```bash
   python -m PyInstaller --clean llama-cpp-launcher.spec --noconfirm
   ```

### Spec 文件说明 (`llama-cpp-launcher.spec`)
| 参数 | 作用 |
|------|------|
| `console=False` | 无控制台窗口（GUI模式） |
| `upx=True` | UPX压缩，减小exe体积 |
| `runtime_tmpdir=None` | 使用默认临时目录解压资源 |

### 打包注意事项
- **单实例保护**：程序已内置 Windows Mutex 机制，双击多次只会打开一个窗口
- **BASE_DIR 处理**：通过 Windows API 获取父进程路径（PyInstaller onefile 模式下 `sys.executable` 指向临时目录）
- **首次运行**：确保 exe 同级目录下有 `config.yml` 配置文件

### 依赖清单 (`requirements.txt`)
| 包名 | 用途 |
|------|------|
| `customtkinter>=5.0.0` | 现代化 GUI 框架 |
| `pyyaml>=6.0` | YAML 配置读写 |
| `nvidia-ml-py` | NVIDIA GPU 监控 |
| `psutil>=5.9.0` | CPU/内存监控系统资源 |
| `py-cpuinfo>=1.0.5` | 获取处理器型号信息 |
| **打包依赖** ||
| `PyInstaller>=6.0.0` | Python → EXE 打包工具 |

### 运行打包后的 exe
- `llama-cpp-launcher.exe` 启动后自动在 **exe 同级目录** 下读取和写入 `config.yml`
- 首次运行时无配置文件，手动设置参数后点击保存即可生成