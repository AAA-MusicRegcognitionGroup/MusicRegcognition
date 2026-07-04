# 基于哼唱的听歌识曲系统

《信号与系统》课程大作业。核心技术路线：自相关法提取基频 → 滑动窗局部归一化 + 子序列 DTW 匹配。

## 快速开始

```bash
pip install -r requirements.txt    # 安装依赖
python process_midi.py             # 预处理 MIDI 曲库（首次或曲库变更后必须执行）
python web_app.py                  # Web 界面 http://127.0.0.1:5000
# 或者
python demo.py                     # 命令行交互
```

## 运行入口

| 入口 | 说明 |
|---|---|
| `web_app.py` | Web 界面。上传音频/浏览器录音 → DTW 检索 → Top-10 排名 + 匹配折线图。支持 14 首批量评测 + 人声分离开关 |
| `demo.py` | 命令行交互。模式 1：单首评测 + 折线图；模式 2：录音 → 可选人声分离 → DTW 检索 → Top-10 + MIDI 试听 |
| `match_live.py` | 直接对 `进阶1/live_test.wav` 检索，画双面板比对图 |
| `batch_evaluate.py` | 批量评测 14 首，打印严格/八度容错准确率 + 平均半音误差表格 |

## 架构

```
record_audio.py ──→ demo_samples/进阶1/
       ↑
demo_samples/task2/*.wav ──→ process_audio.py ──→ dtw_compare.py
       ↑                          ↑                      ↑
  MIR-QBSH 数据集           extract_pitch.py        midi_cache/*.npy
                           (自相关单帧提取)         (MIDI 预计算缓存)
```

## 核心模块

### extract_pitch.py — 单帧基频提取

自相关法，搜索范围 50–1000 Hz，支持 Hamming/Blackman 窗。无效帧返回 0。

```python
extract_pitch(frame, fs, window='hamming')
```

### process_audio.py — 音频级基频序列

分帧 → 60Hz 高通滤波去低频噪声 → RMS 能量 VAD（默认自适应阈值 `median×0.4`, 下限 0.015）→ 单帧提取 → Hz→MIDI → 仅对发声帧做 kernel_size=11 中值滤波。静音帧置 0。

```python
get_pitch_contour(audio_path, sr=8000, frame_samples=256, hop_samples=256,
                  energy_threshold='auto', apply_median=True)
```

### dtw_compare.py — 子序列 DTW 匹配（核心）

相比传统 DTW 做了 5 项针对性优化：

1. **二次中值滤波**（kernel_size=7）：消除自相关倍频/半频残余突刺
2. **差异化静音处理**：只剔除哼唱静音帧，保留 MIDI 完整时间轴
3. **音符量化**（`np.round`）：浮点音高归到整数半音
4. **滑动窗局部零均值归一化**（窗口 350 帧 ≈ 11.2s）：对 MIDI 序列逐帧减去周围发音帧的局部均值。相比全局归一化，长曲不同段落不会互相污染；休止符不参与窗口均值计算，归一化后休止符自动变成大负数，阻止 DTW 误匹配到空白段
5. **子序列 DTW**（`subseq=True`）：哼唱片段可匹配完整乐曲的任意中间段

```python
calculate_similarity(seq1, seq2, remove_silence=True, local_window=350)
```

返回路径长度归一化距离，越小越匹配。

### process_midi.py — MIDI 预处理

遍历 `demo_samples/midiFile/*.mid`，按 32ms 步长转为 pitch contour，存入 `midi_cache/*.npy`。步长必须与音频处理一致。

### record_audio.py — 麦克风录制

PyAudio 录音，8000 Hz / 16bit / mono，自动音量归一化。支持定长和手动停止两种模式。

### split_demo.py — 人声分离

调用 audio-separator 模型，自动从内置模型列表中匹配最佳可用模型（优先 RoFormer → MDX-Net → 保底），将带伴奏音频分离为人声和伴奏。

## 目录结构

```
demo_samples/
  task2/00001-00014.{wav,pv}   # 14 对 MIR-QBSH 测试样本 + 人工标注
  进阶1/                        # 录音/上传音频输出
  midiFile/00001-00014.mid     # 14 首标准乐谱
midi_cache/00001-00014.npy      # 14 首预计算 pitch contour
templates/                      # Web 前端模板
```

数据集来自 MIR-QBSH（MIREX 官方 QBSH 评测集），不在仓库内，完整数据集位于 `../MIR-QBSH/`。

## 关键约定

- **静音帧 = 0**，所有下游逻辑自动跳过
- **采样率固定 8000 Hz，帧长=帧移=256 samples（32ms）**——所有模块必须一致，错位导致准确率断崖下降
- `.pv` 标注与音频帧对齐同一 32ms 步长
- **MIDI 局部归一化时仅统计窗口内发音帧（>0）均值**——这是阻止休止符误匹配的核心机制
- 所有脚本从项目根目录运行
