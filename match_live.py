import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from process_audio import get_pitch_contour
from dtw_compare import calculate_similarity


def main():
    wav_path = "demo_samples/进阶1/live_test.wav"
    cache_dir = "midi_cache/"

    if not os.path.exists(wav_path):
        print(f"错误: 找不到录音文件 {wav_path}")
        return

    npy_files = sorted(glob.glob(os.path.join(cache_dir, "*.npy")))
    if not npy_files:
        print(f"错误: {cache_dir} 中没有缓存文件，请先运行 process_midi.py")
        return

    print("提取哼唱基频...")
    live_seq = get_pitch_contour(wav_path, sr=8000, frame_samples=256, hop_samples=256)

    print(f"正在与 {len(npy_files)} 首曲库比对...\n")
    results = []
    for n_file in npy_files:
        song_id = os.path.basename(n_file).replace(".npy", "")
        midi_seq = np.load(n_file)
        score = calculate_similarity(live_seq, midi_seq)
        results.append({"song": song_id, "dist": score})

    results.sort(key=lambda x: x["dist"])

    print("-" * 50)
    for i, res in enumerate(results):
        marker = " <<<" if i == 0 else ""
        print(f"  {i+1:>2} | {res['song']} | {res['dist']:.4f}{marker}")
    print("-" * 50)
    print(f"\n最匹配: 【{results[0]['song']}】")

    # 画图：录音 vs 第一名
    midi_seq = np.load(os.path.join(cache_dir, f"{results[0]['song']}.npy"))

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), dpi=120,
                                    gridspec_kw={'height_ratios': [2, 1]})

    t_live = np.arange(len(live_seq)) * 32 / 1000
    t_midi = np.arange(len(midi_seq)) * 32 / 1000

    plot_live = np.where(live_seq > 0, live_seq, np.nan)
    plot_midi = np.where(midi_seq > 0, midi_seq, np.nan)

    ax1.plot(t_live, plot_live, color='#1A5276', linewidth=1.2, label='实时录音')
    ax1.plot(t_midi, plot_midi, color='#CD6155', linestyle='--', linewidth=1.8,
             alpha=0.8, label=f'{results[0]["song"]} 标准乐谱')

    all_valid = np.concatenate([live_seq[live_seq > 0], midi_seq[midi_seq > 0]])
    if len(all_valid) > 0:
        ax1.set_ylim([np.min(all_valid) - 3, np.max(all_valid) + 3])

    ax1.set_title(f"实时录音 vs {results[0]['song']} 标准乐谱（完整时间轴）", fontsize=13, fontweight='bold')
    ax1.set_ylabel("MIDI Note")
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.5)

    max_t = min(len(live_seq), len(midi_seq)) * 32 / 1000
    zoom_end = min(5, max_t)

    ax2.plot(t_live, plot_live, color='#1A5276', linewidth=1.2)
    ax2.plot(t_midi, plot_midi, color='#CD6155', linestyle='--', linewidth=1.8, alpha=0.8)

    if len(all_valid) > 0:
        ax2.set_ylim([np.min(all_valid) - 3, np.max(all_valid) + 3])
    ax2.set_xlim([0, zoom_end])
    ax2.set_title("前 5 秒放大", fontsize=12)
    ax2.set_xlabel("时间 (秒)")
    ax2.set_ylabel("MIDI Note")
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
