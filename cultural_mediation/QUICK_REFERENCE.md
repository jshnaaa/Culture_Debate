# 快速参考卡片

## 🚀 一行命令

```bash
# 测试单个样本（Llama）
./run_test.sh llama 0

# 测试单个样本（Qwen）
./run_test.sh qwen 0

# 批量测试10个样本（Llama）
./run_batch_test.sh llama 10

# 批量测试10个样本（Qwen）
./run_batch_test.sh qwen 10
```

---

## 📁 核心文件

| 文件 | 用途 |
|------|------|
| `run_test.sh` | 单样本测试脚本 |
| `run_batch_test.sh` | 批量测试脚本 |
| `quick_start.py` | Python主程序 |
| `config/model_paths.json` | 模型配置 |
| `data/country_weights_init.json` | 国家权重 |

---

## 🎯 参数说明

### run_test.sh

```bash
./run_test.sh [MODEL] [SAMPLE_IDX]
```

- `MODEL`: `llama`（默认）或 `qwen`
- `SAMPLE_IDX`: 样本索引（默认：0）

### run_batch_test.sh

```bash
./run_batch_test.sh [MODEL] [NUM_SAMPLES]
```

- `MODEL`: `llama`（默认）或 `qwen`
- `NUM_SAMPLES`: 测试样本数（默认：10）

---

## 📊 答案格式

- `1` = yes (socially acceptable)
- `2` = no (not socially acceptable)
- `3` = neutral (depends on context)

---

## 🔧 常用命令

```bash
# 测试组件
python test_components.py

# 查看GPU使用
watch -n 1 nvidia-smi

# 查看结果
ls results/
cat results/batch_test_llama_10samples.txt

# 后台运行
nohup ./run_batch_test.sh llama 100 > test.log 2>&1 &
```

---

## 💾 模型路径

- **Llama3.1-8B**: `/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct`
- **Qwen3-8B**: `/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-3-8B-Instruct`
- **数据集**: `/root/autodl-fs/normad_merge_gen.json`

---

## ⚡ 性能

- **模型加载**: ~30秒（首次）
- **单样本**: ~10秒
- **内存**: ~16GB VRAM

---

## 🐛 快速排错

```bash
# 检查路径
ls /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct

# 检查权限
chmod +x run_test.sh run_batch_test.sh

# 清理GPU
python -c "import torch; torch.cuda.empty_cache()"

# 查看详细输出
./run_test.sh llama 0 2>&1 | less
```

---

## 📖 完整文档

- `USAGE_GUIDE.md` - 详细使用指南
- `README.md` - 完整文档
- `DESIGN_SUMMARY.md` - 设计方案
