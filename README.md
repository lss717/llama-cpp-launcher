# 🚀 Llama-CPP 启动器 V1.0

一个功能完备的 llama.cpp server 图形化启动工具，支持实时资源监控、多显卡管理、配置持久化。

## ✨ 核心功能

### 🖥️ **图形化界面管理**
- **可视化配置**：无需命令行，通过 GUI 界面快速设置所有 llama.cpp 参数
- **实时命令预览**：界面下方动态显示即将执行的完整命令行，方便核对

### 📊 **全方位系统监控**
- **CPU**：实时显示处理器型号及占用率
- **内存**：实时显示内存使用情况与总量
- **GPU**：支持多显卡实时监控
  - 显卡型号 | GPU利用率 | 显存占用 | 温度 | 功耗

### 🎮 **智能显卡管理**
- **自动识别**：启动时自动检测系统所有 NVIDIA 显卡
- **显卡选择**：下拉菜单选择单卡或"所有显卡 (并行)"
- **主卡权重分配**：多卡模式下自动计算 Tensor Split (`-ts`) 参数
- **环境变量自动设置**：根据选择自动配置 `CUDA_VISIBLE_DEVICE_S`

### ⚡ **启动参数全覆盖**
| 参数 | 说明 |
|------|------|
| `--host` / `--port` | 服务监听地址与端口 |
| `-m` | 模型文件路径 |
| `-mm` | 多模态模型路径 (可选) |
| `-ngl` | GPU 层数 (支持 `all` 或具体数值) |
| `-mg` | 主显卡编号 |
| `-ts` | Tensor Split 分配 |
| `-c` | 上下文长度 (预设或自定义) |
| `-np` | 并发处理数量 |
| `--cache-type-k/v` | KV 缓存量化类型 (支持 f16, q8_0, turbo4 等) |
| `--reasoning` | 思考模式开关 |
| `--flash-attn` | Flash Attention 优化开关 |
| `--perf` | 性能计时输出 |

### 📝 **配置持久化**
- 所有设置自动保存至 `config.yml`
- 下次启动自动加载上次配置
- 支持手动编辑配置文件

### 📋 **运行日志与统计**
- **实时日志**：彩色高亮显示 (信息、警告、错误、llama.cpp 输出)
- **Token 统计**：实时解析并显示：
  - 总生成 Token 数
  - 生成速率 (tokens/s)
  - 上下文占用百分比

---

## 🛠️ 安装与使用

### 环境要求
- **Python 3.8+**
- **NVIDIA 显卡** (可选，仅用于 GPU 监控)

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行
```bash
python llama_launcher_gui.py
```

---

## ⚙️ 配置示例 (`config.yml`)
```yaml
server_path: "D:\Program Files\llama\llama-server.exe"
model_path: "D:\models\model.gguf"
mmproj_path: "D:\models\mmproj.gguf"
host: "0.0.0.0"
port: "8080"
ngl: "all"
ctx: "32768"
ts_ratio: "28"
cache_type: "q8_0"
reasoning: true
```

---

## 🛠️ 启动流程
1. **配置路径**：设置 `llama-server.exe` 和模型文件路径
2. **选择显卡**：选择运行设备，主卡权重将自动计算
3. **启动服务**：点击"🚀 启动服务并保存配置"
4. **实时监控**：查看资源占用与运行日志
5. **停止服务**：点击"🛑 停止服务"强制结束进程

---

## 📄 许可证
MIT License
