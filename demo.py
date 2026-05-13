import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pretty_midi  # 需要 pip install pretty_midi
from process_audio import get_pitch_contour
from dtw_compare import calculate_similarity

# 常见中文字体支持（避免绘图乱码，如果有报错可注释掉这部分）
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def load_pv_file(pv_path):
    """
    读取人工标注的 .pv 文件。
    通常 .pv 文件内可能存储的是 Hz 或 MIDI note。
    我们根据数值大小判断，如果是 Hz 则统一转换为 MIDI 用作比较。
    """
    with open(pv_path, 'r') as f:
        lines = f.readlines()
    
    pv_seq = []
    for line in lines:
        val = float(line.strip())
        if val == 0:
            pv_seq.append(0.0)
        else:
            # 判断是否为 Hz（人声哼唱 MIDI 编号一般 30~90 之间，若超过 120 认为是 Hz）
            if val > 120: 
                pv_seq.append(librosa.hz_to_midi(val))
            else:
                pv_seq.append(val)
    return np.array(pv_seq)

def get_midi_contour(midi_path, hop_ms=10):
    """
    解析 MIDI 文件，并将音符生成与音频时间步长一致的基频序列 (Pitch Contour)
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    end_time = pm.get_end_time()
    hop_sec = hop_ms / 1000.0
    times = np.arange(0, end_time, hop_sec)
    contour = np.zeros_like(times)
    
    for inst in pm.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                start_idx = int(note.start / hop_sec)
                end_idx = int(note.end / hop_sec)
                # 防止由于精度问题导致数组越界
                end_idx = min(end_idx, len(contour))
                contour[start_idx:end_idx] = note.pitch
    return contour

def main():
    wav_path = "demo_samples/waveFile/00020.wav"
    pv_path = "demo_samples/waveFile/00020.pv"
    midi_dir = "demo_samples/midiFile/"
    
    # 确保所需文件都已被用户放入了对应位置
    if not os.path.exists(wav_path) or not os.path.exists(pv_path):
        print("错误: 找不到音频或 .pv 标注文件。请确保路径与文件名正确。")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"尝试寻找的 wav 路径: {os.path.abspath(wav_path)}")
        print(f"尝试寻找的 pv 路径: {os.path.abspath(pv_path)}")
        return
        
    print("====== 任务1：提取基频并对比人工标注 (.pv) ======")
    hop_samples = 256
    sr = 8000
    hop_ms = (hop_samples / sr) * 1000 # 32.0 ms
    
    # 我们自己算法提取的序列
    print("正在使用自相关法提取音频...")
    extracted_seq = get_pitch_contour(wav_path, sr=sr, frame_samples=hop_samples, hop_samples=hop_samples)
    
    # 数据集提供的基准手工序列
    print("正在读取 .pv 文件...")
    pv_seq = load_pv_file(pv_path)
    
    # ------------------ 补充：计算并打印精确的准确率 (%) ------------------
    # 对齐所比较的序列长度，以防因为截断问题长短不一
    min_len = min(len(extracted_seq), len(pv_seq))
    comp_extracted = extracted_seq[:min_len]
    comp_pv = pv_seq[:min_len]
    
    # 【修复重点】：你之前旧代码里 93% 准确率是因为其步长(hop)与 pv 文件严格对齐！
    # MIR-QBSH 数据集的 .pv 是按照采样率 8000Hz、每 256 采样点一帧存放的。
    # 之前的重构版默认为 10ms 步长，导致两者在时间轴上完全错位，所以分数为 6%。
    
    valid_frames = (comp_pv > 0)
    total_valid = np.sum(valid_frames)
    
    if total_valid > 0:
        # np.abs 算偏差 (你原来的 test1.py 里容忍误差是 <= 1.0 个半音)
        diff_abs = np.abs(comp_extracted - comp_pv)
        # 允许八度倍频误差
        correct_frames_strict = np.sum((diff_abs <= 1.0) & valid_frames)
        mod_diff = np.abs((comp_extracted - comp_pv) % 12)
        correct_frames_octave = np.sum(((mod_diff <= 1.0) | (mod_diff >= 11.0)) & (comp_extracted > 0) & valid_frames)
        
        accuracy_strict = (correct_frames_strict / total_valid) * 100
        accuracy_octave = (correct_frames_octave / total_valid) * 100
        
        print(f"--> 基频提取结果统计：")
        print(f"--> 有效总帧数: {total_valid} 帧")
        print(f"--> 绝对准确正确帧数: {correct_frames_strict} 帧")
        print(f"--> 加上八度容错后的正确帧数: {correct_frames_octave} 帧")
        print(f"--> 严格帧级音高提取准确率 (严格): {accuracy_strict:.2f}%")
        print(f"--> 允许高低八度音高提取准确率 (宽容): {accuracy_octave:.2f}%")
        
        diffs = diff_abs[valid_frames]
        # 去掉倍频带来的一两百的极端误差影响后，算真实的旋律偏离
        mean_error = np.mean(mod_diff[valid_frames])
        print(f"--> 平均提取绝对半音误差(排除八度偏差后): {mean_error:.3f} 个半音")
    else:
        print("--> 警告：从 .pv 文件中未找到有效的非静音帧。")
    # ----------------------------------------------------------------------
    
    # 绘制对比折线图
    plt.figure(figsize=(10, 4))
    # 忽略0值为了绘制更漂亮的折线 (将0换为 NaN，Matplotlib画图会自动断开)
    plot_extracted = np.where(extracted_seq > 0, extracted_seq, np.nan)
    plot_pv = np.where(pv_seq > 0, pv_seq, np.nan)

    plt.plot(plot_extracted, label="算法提取提取的基频", color='blue', linewidth=1.5)
    plt.plot(plot_pv, label="人工标注的基准基频 (.pv)", color='red', linestyle='--', linewidth=1.5)
    plt.title("基频提取准确度比对折线图")
    plt.xlabel("时间帧 (帧数)")
    plt.ylabel("音高 (MIDI Note)")
    plt.legend()
    plt.grid(True)
    plt.xlim([0, max(len(plot_extracted), len(plot_pv))])
    
    # plt.show() 放在后面避免阻塞

    print("\n====== 任务2：与曲库中的 MIDI 序列对比相似度 ======")
    midi_files = glob.glob(os.path.join(midi_dir, "*.mid"))
    if not midi_files:
        print("没有在 midiFile/ 文件夹中找到任何 .mid 文件！")
        return
        
    results = []
    
    for m_file in midi_files:
        song_name = os.path.basename(m_file)
        
        # 1. 抽取 MIDI 序列
        midi_seq = get_midi_contour(m_file, hop_ms=hop_ms)
        
        # 2. 与音频提取出的序列经过 DTW 处理比对
        #    因为上面自己写的 calculate_similarity 自带去 0 均值归一化，所以直接扔进去即可
        score = calculate_similarity(extracted_seq, midi_seq)
        
        results.append({
            "song": song_name,
            "dist": score
        })
        
    # 根据距离从小到大排序 (DTW距离越小，相似度越大)
    results.sort(key=lambda x: x["dist"])
    
    print(f"\n【DTW 匹配结果距离表 (从小到大排序)】:")
    print("-" * 40)
    for i, res in enumerate(results):
        print(f"名次 {i+1} | 歌曲: {res['song']:<{20}} | 距离: {res['dist']:.4f}")
    print("-" * 40)
    print(f"★★★ 系统判定最匹配的歌曲是: 【{results[0]['song']}】 ★★★")

    # 最后再展现折线图（如果在程序一开始跑会阻塞后面代码的运行）
    plt.show()

if __name__ == "__main__":
    main()
