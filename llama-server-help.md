# llama-server 命令行参数中文手册

## 1. 通用参数 (Common Params)
* **-h, --help, --usage**: 打印使用说明并退出 [cite: 1]。
* **--version**: 显示版本和构建信息 [cite: 1]。
* **--license**: 显示源代码许可证和依赖项 [cite: 1, 2]。
* **-cl, --cache-list**: 显示缓存中的模型列表 [cite: 2]。
* **--completion-bash**: 打印可 sourcing 的 bash 自动完成脚本 [cite: 1]。
* **-t, --threads N**: 生成期间使用的 CPU 线程数（默认：-1，即自动） [cite: 2]。
* **-tb, --threads-batch N**: 批处理和提示词预处理期间使用的线程数（默认：与 --threads 相同） [cite: 3, 4]。
* **-C, --cpu-mask M**: CPU 亲和性掩码：任意长度的十六进制数，与 --cpu-range 互补（默认：""） [cite: 2]。
* **-Cr, --cpu-range lo-hi**: CPU 亲和性范围，与 --cpu-mask 互补 [cite: 3]。
* **--cpu-strict <0|1>**: 使用严格的 CPU 放置（默认：0） [cite: 3]。
* **--prio N**: 设置进程/线程优先级：low(-1), normal(0), medium(1), high(2), realtime(3)（默认：0） [cite: 3]。
* **--poll <0...100>**: 使用轮询级别等待工作（0 表示不轮询，默认：50） [cite: 3]。
* **-Cb, --cpu-mask-batch M**: 批处理 CPU 亲和性掩码（默认：与 --cpu-mask 相同） [cite: 3]。
* **-Crb, --cpu-range-batch lo-hi**: 批处理 CPU 亲和性范围 [cite: 3]。
* **--cpu-strict-batch <0|1>**: 批处理使用严格的 CPU 放置（默认：与 --cpu-strict 相同） [cite: 3]。
* **--prio-batch N**: 设置批处理进程/线程优先级（默认：0） [cite: 3]。
* **--poll-batch <0|1>**: 使用轮询等待批处理工作（默认：与 --poll 相同） [cite: 3]。
* **-c, --ctx-size N**: 提示词上下文大小（默认：0，表示从模型元数据中加载） [cite: 10, 11]。
* **-n, --predict, --n-predict N**: 待预测的 Token 数量（默认：-1，表示无穷大） [cite: 11]。
* **-b, --batch-size N**: 逻辑最大批处理大小（默认：2048） [cite: 11, 12]。
* **-ub, --ubatch-size N**: 物理最大批处理大小（默认：512） [cite: 12, 13]。
* **--keep N**: 从初始提示词中保留的 Token 数量（默认：0，-1 表示全部） [cite: 13]。
* **--swa-full**: 使用全尺寸的 SWA 缓存（默认：false） [cite: 13, 14]。
* **-fa, --flash-attn [on|off|auto]**: 设置是否使用 Flash Attention 优化（默认：auto） [cite: 15, 16]。
* **--perf, --no-perf**: 是否启用内部 libllama 性能计时（默认：false） [cite: 16]。
* **-e, --escape, --no-escape**: 是否处理转义序列 (\n, \r, \t, \' ", \\)（默认：true） [cite: 16]。

## 2. GPU 与显存优化 (GPU & VRAM)
* **-ngl, --gpu-layers N**: 存储在 VRAM 中的模型层数。可选具体数字、'auto' 或 'all' [cite: 45, 46]。
* **-sm, --split-mode {none,layer,row}**: 多 GPU 拆分模式：
    * `none`: 仅使用单个 GPU [cite: 47]。
    * `layer` (默认): 按层和 KV 缓存拆分 [cite: 47, 48]。
    * `row`: 按行拆分（计算张量并行） [cite: 48]。
* **-ts, --tensor-split N0,N1...**: 卸载到各 GPU 的比例，用逗号分隔（例如 3,1 表示第一块卡承担 75%） [cite: 48, 49, 50]。
* **-mg, --main-gpu INDEX**: 主 GPU 索引，用于处理中间结果或单卡运行 [cite: 50, 51]。
* **-ctk, --cache-type-k TYPE**: K 缓存的数据类型（支持 f32, f16, bf16, q8_0 等，默认 f16） [cite: 28, 29]。
* **-ctv, --cache-type-v TYPE**: V 缓存的数据类型（同上，默认 f16） [cite: 29, 30, 31]。

## 3. 模型加载 (Model Loading)
* **-m, --model FNAME**: 模型文件的本地路径 [cite: 61, 62]。
* **-mu, --model-url MODEL_URL**: 模型下载 URL（默认：未使用） [cite: 63]。
* **-dr, --docker-repo [<repo>/]<model>[:quant]**: Docker Hub 模型仓库。repo 可选，默认为 ai/。quant 可选，默认为:latest [cite: 63]。
* **-hf, --hf-repo REPO[:quant]**: 从 Hugging Face 仓库加载模型。支持自动选择量化版本（如 Q4_K_M） [cite: 65, 66, 67, 68]。
* **-hfd, --hf-repo-draft REPO[:quant]**: 与 --hf-repo 相同，但用于草稿模型（默认：未使用） [cite: 69]。
* **-hff, --hf-file FILE**: 指定 Hugging Face 仓库中的特定文件 [cite: 70, 71]。
* **-hfv, --hf-repo-v REPO[:quant]**: Vocoder 模型的 Hugging Face 仓库（默认：未使用） [cite: 72]。
* **-hffv, --hf-file-v FILE**: Vocoder 模型的 Hugging Face 文件（默认：未使用） [cite: 73]。
* **-hft, --hf-token TOKEN**: Hugging Face 访问令牌（默认：来自 HF_TOKEN 环境变量） [cite: 73]。
* **--mlock**: 强制系统将模型保留在 RAM 中，防止交换到虚拟内存（Swap） [cite: 32, 33]。
* **--mmap, --no-mmap**: 是否内存映射模型。如果禁用 mmap，加载较慢但可能减少页面换出（默认：启用） [cite: 33, 34]。
* **-kv, --kv-offload, -nkvo, --no-kv-offload**: 是否启用 KV 缓存卸载（默认：启用） [cite: 35]。
* **--repack, --no-repack**: 是否启用权重重新打包（默认：启用） [cite: 35]。
* **--no-host**: 绕过主机缓冲区，允许使用额外缓冲区 [cite: 36]。
* **--rpc SERVERS**: RPC 服务器的逗号分隔列表（host:port） [cite: 37]。
* **--list-devices**: 打印可用设备列表并退出 [cite: 38]。
* **-ot, --override-tensor <pattern>=<type>**: 覆盖张量缓冲区类型 [cite: 39]。
* **--fit [on|off]**: 是否调整未设置的参数以适应设备内存（默认：'on'） [cite: 42]。
* **-fitt, --fit-target MiB0,MiB1,...**: --fit 的每设备目标余量，逗号分隔的值，单个值广播到所有设备（默认：1024） [cite: 42]。
* **-fitc, --fit-ctx N**: --fit 选项可设置的最小 ctx 大小（默认：4096） [cite: 43]。

## 4. 采样参数 (Sampling Params)
* **--samplers SAMPLERS**: 生成时使用的采样器顺序，以 `;` 分隔（默认：penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature） [cite: 88]。
* **-s, --seed SEED**: RNG 种子（默认：-1，使用随机种子） [cite: 89]。
* **--sampler-seq, --sampling-seq SEQUENCE**: 采样器的简化序列（默认：edskypmxt） [cite: 89]。
* **--ignore-eos**: 忽略流结束 Token 并继续生成（隐含 --logit-bias EOS-inf） [cite: 90]。
* **--temp, --temperature N**: 温度系数，控制输出随机性（默认：0.80） [cite: 91]。
* **--top-k N**: 仅从概率前 N 的词中采样（默认：40，0 为禁用） [cite: 91, 92]。
* **--top-p N**: 核采样，累计概率达到 N 时停止（默认：0.95，1.0 为禁用） [cite: 92]。
* **--min-p N**: 最小概率阈值采样（默认：0.05，0.0 为禁用） [cite: 92, 93]。
* **--top-nsigma, --top-n-sigma N**: top-n-sigma 采样（默认：-1.00，-1.0 为禁用） [cite: 93]。
* **--xtc-probability N**: xtc 概率（默认：0.00，0.0 为禁用） [cite: 94]。
* **--xtc-threshold N**: xtc 阈值（默认：0.10，1.0 为禁用） [cite: 94]。
* **--typical, --typical-p N**: 局部典型采样，参数 p（默认：1.00，1.0 为禁用） [cite: 94]。
* **--repeat-last-n N**: 用于惩罚的最后 n 个 token（默认：64，0 为禁用，-1 为 ctx_size） [cite: 94]。
* **--repeat-penalty N**: 重复惩罚系数（默认：1.00，即不惩罚） [cite: 94, 95]。
* **--presence-penalty N**: 存在惩罚（默认：0.00，0.0 为禁用） [cite: 95]。
* **--frequency-penalty N**: 频率惩罚（默认：0.00，0.0 为禁用） [cite: 95]。
* **--dry-multiplier N**: DRY 采样乘数（默认：0.00，0.0 为禁用） [cite: 95, 96]。
* **--dry-base N**: DRY 采样基值（默认：1.75） [cite: 96]。
* **--dry-allowed-length N**: DRY 采样允许长度（默认：2） [cite: 96]。
* **--dry-penalty-last-n N**: DRY 最后 n 个 token 的惩罚（默认：-1，0 为禁用，-1 为上下文大小） [cite: 96]。
* **--dry-sequence-breaker STRING**: DRY 采样序列分隔符，清除默认分隔符 ('\n', ':', '"', '*')；使用"none"不使用任何分隔符 [cite: 96]。
* **--adaptive-target N**: adaptive-p: 选择接近此概率的 token（有效范围 0.0 到 1.0；负数=禁用）（默认：-1.00） [cite: 97]。
* **--adaptive-decay N**: adaptive-p: 目标适应的衰减率。较低值更敏感，较高值更稳定（有效范围 0.0 到 0.99）（默认：0.90） [cite: 97]。
* **--dynatemp-range N**: 动态温度范围（默认：0.00，0.0 为禁用） [cite: 102]。
* **--dynatemp-exp N**: 动态温度指数（默认：1.00） [cite: 102]。
* **--mirostat N**: Mirostat 采样（默认：0，禁用；1 为 Mirostat，2 为 Mirostat 2.0） [cite: 102, 104]。
* **--mirostat-lr N**: Mirostat 学习率，参数 eta（默认：0.10） [cite: 104]。
* **--mirostat-ent N**: Mirostat 目标熵，参数 tau（默认：5.00） [cite: 104]。
* **-l, --logit-bias TOKEN_ID(+/-)BIAS**: 修改 token 在补全中出现的可能性 [cite: 104]。
* **--grammar GRAMMAR**: 约束生成的 BNF 类语法 [cite: 105]。
* **--grammar-file FNAME**: 读取语法的文件 [cite: 105]。
* **-j, --json-schema SCHEMA**: 约束生成的 JSON schema [cite: 105]。
* **-jf, --json-schema-file FILE**: 包含 JSON schema 的文件 [cite: 106]。
* **-bs, --backend-sampling**: 启用后端采样（实验性）（默认：禁用） [cite: 106]。

## 5. 服务端设置 (Server Specific)
* **--host HOST**: 监听地址，或以 `.sock` 结尾绑定到 UNIX socket（默认：127.0.0.1） [cite: 135, 136]。
* **--port PORT**: 监听端口（默认：8080） [cite: 136, 137]。
* **--reuse-port**: 允许多个 socket 绑定到同一端口（默认：禁用） [cite: 137]。
* **--path PATH**: 提供静态文件的路径（默认：""） [cite: 138]。
* **--api-prefix PREFIX**: 服务器提供服务的路径前缀，不带末尾斜杠（默认：""） [cite: 139, 140]。
* **--webui-config JSON**: 提供默认 WebUI 设置的 JSON（覆盖 WebUI 默认值） [cite: 140]。
* **--webui-config-file PATH**: 提供默认 WebUI 设置的 JSON 文件 [cite: 141]。
* **--webui-mcp-proxy, --no-webui-mcp-proxy**: 实验性：是否启用 MCP CORS 代理 - 不要在不受信任的环境中启用（默认：禁用） [cite: 141]。
* **-np, --parallel N**: 服务器并行插槽数量，支持多用户并发（默认：-1，-1 = auto） [cite: 121, 122]。
* **-cb, --cont-batching, -nocb, --no-cont-batching**: 是否启用连续批处理（又称动态批处理，默认：启用） [cite: 122, 123]。
* **--webui, --no-webui**: 是否启用 Web UI（默认：启用） [cite: 145, 146]。
* **--embedding, --embeddings**: 仅支持嵌入用例；仅与专用嵌入模型一起使用（默认：禁用） [cite: 146]。
* **--rerank, --reranking**: 启用服务器上的重排序端点（默认：禁用） [cite: 147]。
* **--api-key KEY**: 用于身份验证的 API 密钥，支持以逗号分隔提供多个密钥（默认：none） [cite: 148, 149]。
* **--api-key-file FNAME**: 包含 API 密钥的文件路径（默认：none） [cite: 149]。
* **--ssl-key-file FNAME**: PEM 编码的 SSL 私钥文件路径 [cite: 150]。
* **--ssl-cert-file FNAME**: PEM 编码的 SSL 证书文件路径 [cite: 151]。
* **--chat-template-kwargs STRING**: 为 json 模板解析器设置附加参数，必须是有效的 json 对象字符串 [cite: 152]。
* **-to, --timeout N**: 服务器读/写超时时间（秒）（默认：600） [cite: 153]。
* **--threads-http N**: 用于处理 HTTP 请求的线程数（默认：-1） [cite: 153]。
* **--cache-prompt, --no-cache-prompt**: 是否启用提示词缓存（默认：启用） [cite: 154]。
* **--cache-reuse N**: 尝试通过 KV 移位从缓存重用的最小块大小，需要启用提示词缓存（默认：0） [cite: 154]。
* **--metrics**: 启用与 Prometheus 兼容的指标端点（默认：禁用） [cite: 156]。
* **--props**: 启用通过 POST /props 更改全局属性（默认：禁用） [cite: 156]。
* **--slots, --no-slots**: 公开插槽监控端点（默认：启用） [cite: 157, 158]。
* **--slot-save-path PATH**: 保存插槽 kv 缓存的路径（默认：禁用） [cite: 158]。
* **--media-path PATH**: 加载本地媒体文件的目录；可通过 file:// URL 使用相对路径访问文件（默认：禁用） [cite: 158]。
* **--models-dir PATH**: 包含路由器服务器模型的目录（默认：禁用） [cite: 159]。
* **--models-preset PATH**: 包含路由器服务器模型预设的 INI 文件路径（默认：禁用） [cite: 159]。
* **--models-max N**: 对于路由器服务器，同时加载的最大模型数（默认：4，0 = 无限制） [cite: 160]。
* **--models-autoload, --no-models-autoload**: 对于路由器服务器，是否自动加载模型（默认：启用） [cite: 160]。
* **--jinja, --no-jinja**: 是否对聊天使用 jinja 模板引擎（默认：启用） [cite: 161]。
* **--reasoning-format FORMAT**: 控制响应中`<think>`标签的处理。可选：`none`、`deepseek`、`deepseek-legacy`（默认：auto） [cite: 164, 165, 166, 167]。
* **-rea, --reasoning [on|off|auto]**: 在聊天中使用推理/思考（默认：'auto' 从模板检测） [cite: 168]。
* **--reasoning-budget N**: 思考的 token 预算：-1 为不限制，0 为立即结束，N>0 为 token 预算（默认：-1） [cite: 169, 170]。
* **--reasoning-budget-message MESSAGE**: 推理预算耗尽时注入到 end-of-thinking 标签之前的消息（默认：none） [cite: 170]。
* **--chat-template JINJA**: 设置自定义 Jinja 聊天模板 [cite: 171, 172]。
* **--chat-template-file JINJA_TEMPLATE_FILE**: 设置自定义 Jinja 聊天模板文件 [cite: 176]。
* **--skip-chat-parsing, --no-skip-chat-parsing**: 强制使用纯内容解析器，即使指定了 Jinja 模板（默认：禁用） [cite: 180]。
* **--prefill-assistant, --no-prefill-assistant**: 如果最后一条消息是助手消息，是否预填充助手的响应（默认：预填充启用） [cite: 181]。
* **-sps, --slot-prompt-similarity SIMILARITY**: 请求的提示词与插槽的提示词匹配程度（默认：0.10，0.0 = 禁用） [cite: 184]。
* **--lora-init-without-apply**: 加载 LoRA 适配器但不应用它们（默认：禁用） [cite: 184]。
* **--sleep-idle-seconds SECONDS**: 服务器进入睡眠状态的空闲秒数（默认：-1；-1 = 禁用） [cite: 185]。

---

### 6. 特定示例与高级参数 (Example-specific Params)

#### 缓存与解码
* **-lcs, --lookup-cache-static FNAME**: 用于查找解码的静态查找缓存路径（不由生成更新） [cite: 108]。
* **-lcd, --lookup-cache-dynamic FNAME**: 用于查找解码的动态查找缓存路径（由生成更新） [cite: 109]。
* **-ctxcp, --ctx-checkpoints, --swa-checkpoints N**: 每插槽创建的最大上下文检查点数量（默认：32） [cite: 110]。
* **-cpent, --checkpoint-every-n-tokens N**: 预填充（处理）期间每 n 个 token 创建一个检查点，-1 禁用（默认：8192） [cite: 112]。
* **-cram, --cache-ram N**: 设置最大缓存大小（MiB）（默认：8192，-1 无限制，0 禁用） [cite: 113]。
* **-kvu, --kv-unified, -no-kvu, --no-kv-unified**: 使用所有序列共享的单个统一 KV 缓冲区（默认：如果插槽数为 auto 则启用） [cite: 115]。
* **--context-shift, --no-context-shift**: 是否在无限文本生成中使用上下文移位（默认：禁用） [cite: 116]。
* **-r, --reverse-prompt PROMPT**: 在 PROMPT 处停止生成，在交互模式下返回控制 [cite: 117]。
* **-sp, --special**: 特殊 token 输出已启用（默认：false） [cite: 117]。
* **--warmup, --no-warmup**: 是否通过空运行执行预热（默认：启用） [cite: 118]。
* **--spm-infill**: 对 infill 使用 Suffix/Prefix/Middle 模式（而不是 Prefix/Suffix/Middle）（默认：禁用） [cite: 118]。
* **--pooling {none,mean,cls,last,rank}**: 嵌入的 pooling 类型，如果未指定则使用模型默认值 [cite: 119]。

#### 并行与批处理
* **-np, --parallel N**: 服务器插槽（Slots）数量，决定支持的并发请求数（默认：-1，自动） [cite: 121, 122]。
* **-cb, --cont-batching, -nocb, --no-cont-batching**: 是否启用连续批处理（又称动态批处理，默认：启用） [cite: 122, 123]。

#### 多模态 (Multimodal)
* **-mm, --mmproj FILE**: 多模态投影器（Projector）文件路径 [cite: 123, 124]。
* **-mmu, --mmproj-url URL**: 多模态投影器文件的 URL [cite: 125]。
* **--mmproj-auto, --no-mmproj, --no-mmproj-auto**: 是否使用多模态投影器文件（如果可用），在使用 -hf 时有用（默认：启用） [cite: 126]。
* **--mmproj-offload, --no-mmproj-offload**: 是否将多模态投影器卸载到 GPU（默认：启用） [cite: 127, 128]。
* **--image-min-tokens N**: 每个图像可以占用的最小 token 数，仅用于具有动态分辨率的视觉模型（默认：从模型读取） [cite: 128, 129]。
* **--image-max-tokens N**: 每个图像可以占用的最大 token 数，仅用于具有动态分辨率的视觉模型（默认：从模型读取） [cite: 129]。

#### 模型元数据
* **-a, --alias STRING**: 设置模型名称别名，逗号分隔（由 API 使用） [cite: 131]。
* **--tags STRING**: 设置模型标签，逗号分隔（信息性，不用于路由） [cite: 132]。

#### LoRA 与控制向量
* **--lora FNAME**: LoRA 适配器的路径（使用逗号分隔值加载多个适配器） [cite: 269]。
* **--lora-scaled FNAME:SCALE,...**: 带有用户定义缩放的 LoRA 适配器路径（格式：FNAME:SCALE,...） [cite: 270]。
* **--control-vector FNAME**: 添加控制向量（使用逗号分隔值添加多个） [cite: 271]。
* **--control-vector-scaled FNAME:SCALE,...**: 添加带有用户定义缩放 SCALE 的控制向量 [cite: 272]。
* **--control-vector-layer-range START END**: 应用控制向量的层范围，包含起点和终点 [cite: 273]。

---

### 7. 网络与服务配置 (Network & API Config)

* **--host HOST**: 监听的 IP 地址。如果地址以 `.sock` 结尾，则绑定到 UNIX 套接字（默认：127.0.0.1） [cite: 136]。
* **--port PORT**: 监听端口（默认：8080） [cite: 137]。
* **--path PATH**: 提供静态文件的路径 [cite: 138]。
* **--api-prefix PREFIX**: 服务器提供服务的路径前缀，不带末尾斜杠 [cite: 139, 140]。
* **--api-key KEY**: 用于身份验证的 API 密钥，支持以逗号分隔提供多个密钥 [cite: 149]。
* **--ssl-key-file FNAME**: PEM 编码的 SSL 私钥文件路径 [cite: 150]。
* **--ssl-cert-file FNAME**: PEM 编码的 SSL 证书文件路径 [cite: 151]。
* **--timeout N**: 服务器读/写超时时间（秒）（默认：600） [cite: 153]。
* **--threads-http N**: 用于处理 HTTP 请求的线程数（默认：-1） [cite: 153]。

### 8. 日志与监控 (Logging & Metrics)

* **--log-disable**: 禁用日志记录 [cite: 74]。
* **--log-file FNAME**: 将日志输出到指定文件 [cite: 75]。
* **--log-colors [on|off|auto]**: 设置彩色日志（默认：auto，仅在终端输出时启用） [cite: 75, 76]。
* **-v, --verbose, --log-verbose**: 设置详细程度为无穷大，记录所有消息，适用于调试 [cite: 76, 77]。
* **--offline**: 离线模式：强制使用缓存，阻止网络访问 [cite: 77]。
* **-lv, --verbosity, --log-verbosity N**: 设置详细程度阈值。值：0-通用输出，1-错误，2-警告，3-信息，4-调试（默认：3） [cite: 78]。
* **--log-prefix**: 在日志消息中启用前缀 [cite: 83]。
* **--log-timestamps**: 在日志消息中启用时间戳 [cite: 83, 84]。
* **--metrics**: 启用与 Prometheus 兼容的指标端点（默认：禁用） [cite: 156]。
* **--props**: 启用通过 POST /props 更改全局属性（默认：禁用） [cite: 156]。
* **--slots, --no-slots**: 公开插槽监控端点（默认：启用） [cite: 157, 158]。
* **--slot-save-path PATH**: 保存插槽 kv 缓存的路径（默认：禁用） [cite: 158]。

### 9. 投机采样与草稿模型 (Speculative Decoding)

* **-td, --threads-draft N**: 生成期间使用的线程数（默认：与 --threads 相同） [cite: 192]。
* **-tbd, --threads-batch-draft N**: 批处理和提示词处理期间使用的线程数（默认：与 --threads-draft 相同） [cite: 193]。
* **-ctkd, --cache-type-k-draft TYPE**: 草稿模型的 K 的 KV 缓存数据类型（默认：f16） [cite: 193]。
* **-ctvd, --cache-type-v-draft TYPE**: 草稿模型的 V 的 KV 缓存数据类型（默认：f16） [cite: 193]。
* **-md, --model-draft FNAME**: 用于投机采样的草稿模型路径 [cite: 201]。
* **--draft, --draft-max N**: 投机采样生成的草稿 Token 数量（默认：16） [cite: 194]。
* **--draft-min, --draft-n-min N**: 投机采样使用的最小草稿 Token 数量（默认：0） [cite: 195]。
* **--draft-p-min P**: 投机采样的最小概率阈值（贪婪采样）（默认：0.75） [cite: 196]。
* **-cd, --ctx-size-draft N**: 草稿模型的提示词上下文大小（默认：0，从模型加载） [cite: 196, 197]。
* **-devd, --device-draft <dev1,dev2,..>**: 卸载草稿模型的设备列表（逗号分隔） [cite: 198]。
* **-ngld, --gpu-layers-draft, --n-gpu-layers-draft N**: 存储在 VRAM 中的草稿模型层数（默认：auto） [cite: 199, 200]。
* **-otd, --override-tensor-draft <pattern>=<type>**: 覆盖草稿模型的张量缓冲区类型 [cite: 200]。
* **-cmoed, --cpu-moe-draft**: 将草稿模型的所有 MoE 权重保留在 CPU [cite: 200]。
* **-ncmoed, --n-cpu-moe-draft N**: 将草稿模型的前 N 层 MoE 权重保留在 CPU [cite: 201]。
* **--spec-replace TARGET DRAFT**: 如果草稿模型和主模型不兼容，将 TARGET 中的字符串转换为 DRAFT [cite: 204]。
* **--spec-type [none|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod]**: 未提供草稿模型时使用的投机采样类型（默认：none） [cite: 202, 203]。
* **--spec-ngram-size-n N**: ngram-simple/ngram-map 投机采样的 ngram 大小 N，查找 n-gram 的长度（默认：12） [cite: 203, 204]。
* **--spec-ngram-size-m N**: ngram-simple/ngram-map 投机采样的 ngram 大小 M，草稿 m-gram 的长度（默认：48） [cite: 204]。
* **--spec-ngram-min-hits N**: ngram-map 投机采样的最小命中次数（默认：1） [cite: 205]。

### 10. 智能体与工具 (Tools & Agents - 实验性)

* **--tools TOOL1,TOOL2,...**: 启用内置 AI 智能体工具（默认：无工具）。指定"all"启用所有工具。可用工具：`read_file`, `file_glob_search`, `grep_search`, `exec_shell_command`, `write_file`, `edit_file`, `apply_diff` [cite: 143, 144, 145]。
    * *注意：请勿在不受信任的环境中启用此功能* [cite: 143]。

### 11. 投机采样预设与 TTS

* **--mv, --model-vocoder FNAME**: 音频生成的 vocoder 模型（默认：未使用） [cite: 207]。
* **--tts-use-guide-tokens**: 使用指导 token 提高 TTS 单词召回率 [cite: 207]。
* **--embd-gemma-default**: 使用默认 EmbeddingGemma 模型（注意：可以从互联网下载权重） [cite: 208]。
* **--fim-qwen-1.5b-default**: 使用默认 Qwen 2.5 Coder 1.5B（注意：可以从互联网下载权重） [cite: 208]。
* **--fim-qwen-3b-default**: 使用默认 Qwen 2.5 Coder 3B（注意：可以从互联网下载权重） [cite: 209]。
* **--fim-qwen-7b-default**: 使用默认 Qwen 2.5 Coder 7B（注意：可以从互联网下载权重） [cite: 209]。
* **--fim-qwen-7b-spec**: 使用 Qwen 2.5 Coder 7B + 0.5B 草稿进行投机采样（注意：可以从互联网下载权重） [cite: 210]。
* **--fim-qwen-14b-spec**: 使用 Qwen 2.5 Coder 14B + 0.5B 草稿进行投机采样（注意：可以从互联网下载权重） [cite: 210]。
* **--fim-qwen-30b-default**: 使用默认 Qwen 3 Coder 30B A3B Instruct（注意：可以从互联网下载权重） [cite: 211]。
* **--gpt-oss-20b-default**: 使用 gpt-oss-20b（注意：可以从互联网下载权重） [cite: 211]。
* **--gpt-oss-120b-default**: 使用 gpt-oss-120b（注意：可以从互联网下载权重） [cite: 212]。
* **--vision-gemma-4b-default**: 使用 Gemma 3 4B QAT（注意：可以从互联网下载权重） [cite: 212]。
* **--vision-gemma-12b-default**: 使用 Gemma 3 12B QAT（注意：可以从互联网下载权重） [cite: 213]。

---

### 12. RoPE 频率与上下文缩放 (RoPE & Context Scaling)

* **--rope-scaling {none,linear,yarn}**: RoPE 频率缩放方法，默认通常为 `linear` [cite: 229]。
* **--rope-scale N**: 上下文缩放因子，将上下文扩展 N 倍 [cite: 230]。
* **--rope-freq-base N**: RoPE 基础频率，用于 NTK 感知缩放，默认从模型加载 [cite: 231]。
* **--rope-freq-scale N**: RoPE 频率缩放因子，将上下文扩展 1/N 倍 [cite: 231]。
* **--yarn-orig-ctx N**: YaRN 方法中的原始训练上下文大小（默认：0 = 模型训练上下文大小） [cite: 233]。
* **--yarn-ext-factor N**: YaRN 外推混合因子（默认：-1.00，0.0 表示完全插值） [cite: 234]。
* **--yarn-attn-factor N**: YaRN 缩放 sqrt(t) 或注意力幅度（默认：-1.00） [cite: 234]。
* **--yarn-beta-slow N**: YaRN 高修正维度或 alpha（默认：-1.00） [cite: 234]。
* **--yarn-beta-fast N**: YaRN 低修正维度或 beta（默认：-1.00） [cite: 234]。

---

### 13. 进阶系统与内存优化 (Advanced System & Memory)

* **--numa TYPE**: 针对 NUMA 系统的优化，可选：`distribute` (均匀分布执行)、`isolate` (仅在启动节点生成线程) 或 `numactl` [cite: 248, 249]。
* **-dev, --device <dev1,dev2,..>**: 用于卸载的设备逗号分隔列表（none = 不卸载） [cite: 248]。
* **-dt, --defrag-thold N**: KV 缓存碎片整理阈值（已弃用） [cite: 41]。
* **-dio, --direct-io, -ndio, --no-direct-io**: 如果可用，使用 DirectIO（默认：禁用） [cite: 246, 247]。
* **--repack, --no-repack**: 是否启用权重重新打包（默认：启用） [cite: 35]。
* **--no-host**: 绕过主机缓冲区，允许使用额外缓冲区 [cite: 36]。
* **--mlock**: 强制系统将模型保留在 RAM 中，防止交换到虚拟内存（Swap） [cite: 32, 33]。
* **--check-tensors**: 检查模型张量数据中的无效值（默认：false） [cite: 266]。

---

### 14. 模型元数据与算子控制 (Metadata & Op Overrides)

* **--override-kv KEY=TYPE:VALUE**: 强制覆盖模型元数据的进阶选项。支持 `int`, `float`, `bool`, `str` 类型 [cite: 267, 268]。
* **--override-tensor <pattern>=<type>**: 覆盖特定张量的缓冲区类型 [cite: 254]。
* **--op-offload, --no-op-offload**: 是否将主机张量操作卸载到设备（默认：true） [cite: 268]。

---

### 15. 混合专家模型 (MoE) 特有参数

* **-cmoe, --cpu-moe**: 将所有 MoE 专家的权重保留在 CPU 中 [cite: 254]。
* **-ncmoe, --n-cpu-moe N**: 仅将前 N 层的 MoE 权重保留在 CPU 中 [cite: 255, 256]。
* **-cmoed, --cpu-moe-draft**: 将草稿模型的所有 MoE 权重保留在 CPU [cite: 200]。
* **-ncmoed, --n-cpu-moe-draft N**: 将草稿模型的前 N 层 MoE 权重保留在 CPU [cite: 201]。
