# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于哼唱的听歌识曲系统——《信号与系统》课程大作业。核心技术路线：自相关法提取基频 → 子序列 DTW 匹配。

## 运行环境与依赖

```bash
pip install -r requirements.txt
```

`requirements.txt`：numpy, scipy, librosa, matplotlib, pretty_midi

可选依赖：`pyaudio`（录音模块需用），`pygame`（MIDI 播放需用）。

## 快速运行

```bash
# 模式1：数据集评测 (指定1-14号歌曲，含.pv准确率对比+折线图)
python demo.py  # 选 1，输入歌曲编号如 1

# 模式2：现场录唱实时识别 (自动录音→子序列DTW检索→可选MIDI播放)
python demo.py  # 选 2，按提示哼唱

# 预处理 MIDI 曲库为 .npy 缓存 (新加歌曲后需重跑)
python process_midi.py
```

## 架构

```
record_audio.py ─→ 录音保存至 demo_samples/进阶1/
       ↑
demo_samples/task2/*.wav ─→ process_audio.py ─→ dtw_compare.py
       ↑                          ↑                  ↑
  MIR-QBSH 数据集           extract_pitch.py      midi_cache/*.npy
                           (自相关法单帧提取)      (预计算曲库)
```

## 核心模块

### extract_pitch.py — 单帧基频提取
`extract_pitch(frame, fs, window)` — 自相关法，搜索范围 50-1000Hz，支持 Hamming/Blackman 窗。返回 Hz，无有效基频返回 0。

### process_audio.py — 音频级基频序列
`get_pitch_contour(audio_path, sr=8000, frame_samples=256, hop_samples=256, energy_threshold=0.06)` — 分帧 + RMS 能量 VAD + 单帧基频提取 + Hz→MIDI 转换 + `kernel_size=11` 中值滤波。静音帧置 0。

### dtw_compare.py — 子序列 DTW 匹配（核心升级）
`calculate_similarity(seq1, seq2)` — 当前版本包含 5 项关键优化：
1. **二次中值滤波**（kernel_size=7）：进一步抹平自相关倍频/半频突刺
2. **差异化静音处理**：只剔除哼唱的 0，保留 MIDI 完整时间轴
3. **音符量化**（`np.round`）：将浮点音高归到整数半音，消除嗓音自然抖动
4. **改进型零均值归一化**：MIDI 均值只统计发声音符（>0），休止符不参与均值计算，从而休止符变成很大的负数，阻止 DTW 误匹配到空白段
5. **子序列 DTW**（`subseq=True`）：允许哼唱片段匹配完整长曲的任意中间段

返回路径长度归一化 DTW 距离，越小越匹配。

### process_midi.py — MIDI 预处理
`batch_preprocess()` — 遍历 `demo_samples/midiFile/*.mid`，Hop 设为 `32ms`（=256/8000），存为 `midi_cache/*.npy`。**运行必须和音频处理步长一致。**

### record_audio.py — 麦克风录制
`record_audio_to_project(file_name, duration)` — PyAudio 录制，自动 8000Hz/16bit/int16 音量归一化，输出到 `demo_samples/进阶1/`。

### demo.py — 整合入口
交互式选择两种模式：
- **模式1**：选择编号（1-14），加载 `task2/` 中的 .wav 和 .pv，输出严格/八度容错准确率 + matplotlib 折线图
- **模式2**：自动调用录音 → 子序列 DTW 检索 `midi_cache/` 中 14 首曲库 → 打印 Top-10 排名 → 可选 pygame/系统播放器播放第一候选 MIDI

## 目录结构

```
demo_samples/
  task2/00001-00014.{wav,pv}  # 14对 MIR-QBSH 测试样本+人工标注
  进阶1/live_test.wav          # 模式2录音输出路径
  midiFile/00001-00014.mid    # 14首标准乐谱
midi_cache/00001-00014.npy     # 14首预计算 pitch contour
```

## 数据集

MIR-QBSH 为核心数据集（MIREX 官方 QBSH 评测集），不在仓库内，应放在 `../MIR-QBSH/`。`task2/` 中的 14 对 .wav/.pv 即来自该数据集。

## 关键约定

- 静音帧 = 0，所有下游逻辑自动跳过
- 采样率固定 8000Hz，Hop 固定 256 samples = 32ms
- `.pv` 标注文件和音频帧对齐此步长，错误步长会导致准确率断崖下降
- `process_midi.py` 的 Hop 必须与音频处理一致，否则曲库缓存和哼唱序列时序不同
- DTW 前零均值归一化时 MIDI 均值只统计 >0 帧 — 这是阻塞休止符误匹配的核心机制
- 项目根目录运行所有脚本，`record_audio.py` 输出也走根目录相对路径
