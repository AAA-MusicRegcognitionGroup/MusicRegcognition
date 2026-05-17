import os
import glob
import numpy as np
import pretty_midi  # 确保已安装: pip install pretty_midi

# ==========================================
# 1. 从 demo.py 迁移过来的核心解析函数
# ==========================================
def get_midi_contour(midi_path, hop_ms=32.0):
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

# ==========================================
# 2. 自动化遍历与保存逻辑
# ==========================================
def batch_preprocess():
    # 配置参数：必须与音频处理步长一致 (256/8000 = 32ms)
    HOP_MS = 32.0 
    MIDI_DIR = "demo_samples/midiFile/"
    CACHE_DIR = "midi_cache/"

    # 创建缓存目录
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(f"已创建目录: {CACHE_DIR}")

    # 获取所有 MIDI 文件
    midi_files = glob.glob(os.path.join(MIDI_DIR, "*.mid"))
    if not midi_files:
        print(f"未在 {MIDI_DIR} 找到 .mid 文件")
        return

    print(f"开始预处理 {len(midi_files)} 个 MIDI 文件...")

    for m_file in midi_files:
        file_name = os.path.basename(m_file)
        song_id = file_name.replace(".mid", "")
        save_path = os.path.join(CACHE_DIR, f"{song_id}.npy")

        try:
            # 转换并保存
            midi_seq = get_midi_contour(m_file, hop_ms=HOP_MS)
            np.save(save_path, midi_seq)
            print(f"成功导出: {song_id}.npy")
        except Exception as e:
            print(f"处理 {file_name} 失败: {e}")

    print("-" * 30)
    print("预处理全部完成！")

if __name__ == "__main__":
    batch_preprocess()