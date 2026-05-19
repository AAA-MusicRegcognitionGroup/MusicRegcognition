import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pretty_midi  # 需要 pip install pretty_midi
from process_audio import get_pitch_contour
from dtw_compare import calculate_similarity

try:
    from split_demo import run_audio_separation
    _HAS_AUDIO_SEPARATOR = True
except ImportError:
    _HAS_AUDIO_SEPARATOR = False

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

def find_vocal_file(output_files):
    """从 audio_separator 输出文件列表中定位人声 stem"""
    for f in output_files:
        basename = os.path.basename(f)
        if "(Vocals)" in basename or "vocals" in basename.lower():
            return f
    return None


def main():
    print("==================================================")
    print("          音乐检索系统 (Query By Humming)         ")
    print("==================================================")
    print("请选择运行模式:")
    print("1. 其一：数据集原曲评测模式 (指定1-14编号，含.pv对比与画图)")
    print("2. 其二：现场哼唱实时识别模式 (录音保存至'进阶1'，子序列检索，无画图)")
    print("--------------------------------------------------")
    
    mode = input("请输入选择的模式序号 (1 或 2): ").strip()
    
    # ==============================================================================
    # 部分一：数据集原曲评测模式
    # ==============================================================================
    if mode == "1":
        song_num = input("请输入需要比对的歌曲编号 (1-14): ").strip()
        # 自动补全为数据集标准的5位编号格式（例如输入 11 变为 00011）
        padded_num = song_num.zfill(5)
        
        # 唯一定位到 task2 文件夹下的原有数据集文件
        wav_path = f"demo_samples/task2/{padded_num}.wav"
        pv_path = f"demo_samples/task2/{padded_num}.pv"
        
        # 将文件锁死检查独立在部分一内部，互不干扰
        if not os.path.exists(wav_path) or not os.path.exists(pv_path):
            print("错误: 找不到对应的音频或 .pv 标注文件。请确保路径与文件名正确。")
            print(f"尝试寻找的 wav 路径: {os.path.abspath(wav_path)}")
            print(f"尝试寻找的 pv 路径: {os.path.abspath(pv_path)}")
            return
            
        print("\n====== 任务1：提取基频并对比人工标注 (.pv) ======")
        hop_samples = 256
        sr = 8000
        
        print("正在使用自相关法提取音频...")
        extracted_seq = get_pitch_contour(wav_path, sr=sr, frame_samples=hop_samples, hop_samples=hop_samples)
        
        print("正在读取 .pv 文件...")
        pv_seq = load_pv_file(pv_path)
        
        # 保留原有的五秒/长度截断对齐逻辑
        min_len = min(len(extracted_seq), len(pv_seq))
        comp_extracted = extracted_seq[:min_len]
        comp_pv = pv_seq[:min_len]
        
        valid_frames = (comp_pv > 0)
        total_valid = np.sum(valid_frames)
        
        if total_valid > 0:
            diff_abs = np.abs(comp_extracted - comp_pv)
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
            
            mean_error = np.mean(mod_diff[valid_frames])
            print(f"--> 平均提取绝对半音误差(排除八度偏差后): {mean_error:.3f} 个半音")
        else:
            print("--> 警告：从 .pv 文件中未找到有效的非静音帧。")
        
        # 保留画图逻辑
        plot_extracted = np.where(extracted_seq > 0, extracted_seq, np.nan)
        plot_pv = np.where(pv_seq > 0, pv_seq, np.nan)

        # 更换更高级的红蓝配色，并使用半透明填充空隙
        plt.figure(figsize=(11, 4.5), dpi=120) # 提高分辨率

        # 绘制线条，调整粗细与层级
        plt.plot(plot_extracted, label="算法提取的基频", color='#1A5276', linewidth=1.5, zorder=2)
        plt.plot(plot_pv, label="人工标注的基准基频 (.pv)", color='#CD6155', linestyle='--', linewidth=2.0, zorder=1)

        # 美化坐标轴和网格
        plt.title("基频提取准确度比对折线图", fontsize=14, pad=15, fontweight='bold')
        plt.xlabel("时间帧 (帧数)", fontsize=11, labelpad=8)
        plt.ylabel("音高 (MIDI Note)", fontsize=11, labelpad=8)
        plt.grid(True, linestyle=':', alpha=0.6, color='gray') # 吧实线网格改成高级的细虚线

        # 调整刻度和留白，防止线条贴边
        plt.tick_params(direction='in', top=True, right=True) # 刻度线朝内
        plt.xlim([-5, len(plot_extracted) + 5]) # 左右留出 5 帧的呼吸空间
        
        # 过滤掉全零的静音帧，只根据有效的音高数据动态计算坐标轴上下限
        valid_pitches = pv_seq[pv_seq > 0]
        if len(valid_pitches) > 0:
            plt.ylim([np.min(valid_pitches) - 2, np.max(valid_pitches) + 2])
        else:
            plt.ylim([40, 80]) # 防御性备用边界

        plt.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True, loc='upper right')
        plt.tight_layout()
        
        print("\n>>> 模式1评测完成，正在展示对比图表...")
        plt.show()

    # ==============================================================================
    # 部分二：现场哼唱实时识别模式
    # ==============================================================================
    elif mode == "2":
        # 确保新创建的进阶1目录存在
        target_dir = "demo_samples/进阶1"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        wav_path = os.path.join(target_dir, "live_test.wav")
        
        print("\n>>> 正在唤醒录音程序，开始录制您的哼唱...")
        # 级联唤醒外部录音脚本
        if os.path.exists("record_audio.py"):
            os.system("python record_audio.py")
        elif os.path.exists("record_to_wavefile.py"):
            os.system("python record_to_wavefile.py")
        else:
            print("提示: 未在根目录下找到自动录音脚本。")
            print(f"请手动将录好的音频命名为 live_test.wav 并放入: {os.path.abspath(target_dir)}")
            input("确认文件就绪后，按回车键继续程序...")

        if not os.path.exists(wav_path):
            print(f"错误: 未能检测到录音音频文件: {wav_path}")
            return
            
        print("\n>>> 录音读取成功！")

        # 尝试使用 AI 人声分离提升匹配精度
        vocal_wav_path = wav_path  # 默认回退到原始录音
        if _HAS_AUDIO_SEPARATOR:
            print(">>> 正在进行 AI 人声分离（首次运行需下载模型约300MB，请耐心等待）...")
            try:
                sep_output_dir = target_dir
                output_files = run_audio_separation(wav_path, output_dir=sep_output_dir)
                vocal_file = find_vocal_file(output_files)
                if vocal_file and os.path.exists(vocal_file):
                    vocal_wav_path = vocal_file
                    print(f">>> 人声分离成功，将使用分离后的人声进行匹配: {os.path.basename(vocal_file)}")
                else:
                    print(">>> 警告：未能从分离输出中定位人声文件，回退使用原始录音。")
            except Exception as e:
                print(f">>> 人声分离失败 ({e})，回退使用原始录音。")
        else:
            print(">>> 提示：未安装 audio-separator，跳过人声分离，直接使用原始录音。")
            print(">>> 如需启用：pip install \"audio-separator[cpu]\"  # CPU 版本")
            print(">>>           pip install \"audio-separator[gpu]\"  # NVIDIA GPU 版本")

        print(">>> 正在提取人声音高特征...")
        hop_samples = 256
        sr = 8000
        # 对分离后的人声（或回退的原始录音）提取完整波形特征，不采用任何 5 秒截断逻辑
        extracted_seq = get_pitch_contour(vocal_wav_path, sr=sr, frame_samples=hop_samples, hop_samples=hop_samples)
        
        print("\n====== 正在利用 子序列 DTW 算法匹配完整检索曲库 ======")
        cache_dir = "midi_cache/" 
        npy_files = glob.glob(os.path.join(cache_dir, "*.npy")) 
        
        if not npy_files:
            print(f"错误：在 {cache_dir} 中没找到缓存文件！请先运行 preprocess_midi.py")
            return
            
        results = []
        for n_file in npy_files:
            song_name = os.path.basename(n_file).replace(".npy", "")
            midi_seq = np.load(n_file) 
            
            # 这里调用的是底层已经变更为子序列匹配(subseq=True)的相似度算法
            score = calculate_similarity(extracted_seq, midi_seq)
            results.append({"song": song_name, "dist": score})    
            
        # 依据子序列代价值从小到大进行精确排序
        results.sort(key=lambda x: x["dist"])
        
        # 严格执行限制：不保留画图，仅打印前十名与最终判定
        print(f"\n【子序列 DTW 最佳匹配结果 (仅展示前10个最小距离)】:")
        print("-" * 50)
        for i, res in enumerate(results[:10]):
            print(f"推荐排位 {i+1} | 候选歌曲: {res['song']:<{20}} | 匹配代价值: {res['dist']:.4f}")
        print("-" * 50)
        print(f"★★★ 检索完成！系统判定您唱的歌曲最有可能是: 【{results[0]['song']}】 ★★★\n")

        # ----------------------------------------------------------------------
        # 新增扩展功能：交互式询问并播放最有可能是的该歌曲 MIDI 文件
        # ----------------------------------------------------------------------
        play_choice = input(f"是否播放最有可能是的歌曲【{results[0]['song']}】的 MIDI 文件？(y/n): ").strip().lower()
        if play_choice == 'y':
            midi_file_dir = "demo_samples/midiFile"
            midi_path = os.path.join(midi_file_dir, f"{results[0]['song']}.mid")
            
            # 兼容处理兼容性可能带来的 .midi 后缀名
            if not os.path.exists(midi_path):
                midi_path = os.path.join(midi_file_dir, f"{results[0]['song']}.midi")
                
            if os.path.exists(midi_path):
                print(f"\n>>> 正在为您播放标准库原曲: {results[0]['song']} ...")
                print(">>> [提示]：在终端中按【Ctrl + C】可随时中断声音播放。")
                try:
                    import pygame
                    pygame.mixer.init()
                    pygame.mixer.music.load(midi_path)
                    pygame.mixer.music.play()
                    # 循环保持程序激活，直到音频播放完毕
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                except ImportError:
                    print("\n[系统提示] 检测到本地未配置 pygame 环境，正在为您尝试调用系统默认外部播放器...")
                    try:
                        if hasattr(os, 'startfile'):
                            os.startfile(midi_path)  # Windows 完美调用默认播放器
                        else:
                            os.system(f"open '{midi_path}'")  # Mac 系统调用
                    except Exception:
                        print("错误：无法唤醒外部播放器，请尝试在控制台手动配置环境: pip install pygame")
                except KeyboardInterrupt:
                    pygame.mixer.music.stop()
                    print("\n>>> 歌曲播放已被用户手动中断。")
            else:
                print(f"\n错误：未在 {midi_file_dir} 文件夹下找到名称为 【{results[0]['song']}.mid】 的标准MIDI。")

    else:
        print("错误输入！选项不合法，程序退出。")

if __name__ == "__main__":
    main()