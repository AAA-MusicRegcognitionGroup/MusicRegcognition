# 音频级别基频序列提取
import numpy as np
import librosa
from scipy.signal import medfilt, butter, sosfiltfilt
from extract_pitch import extract_pitch

# ==========================================
# 调用说明与使用方法
# ==========================================
# 
# 该模块用于读取完整的音频文件（如 .wav），对其进行短时分帧，
# 并调用底层的 extract_pitch 对每一帧求取基频。
# 附带了基于短时能量的静音阈值过滤（置 0 处理），以排除休止符和呼吸杂音。
#
# 示例用法：
"""
    from process_audio import get_pitch_contour
    
    # 提取基频路径，设置窗口 30ms，步长 10ms，能量阈值 0.01
    pitch_contour = get_pitch_contour("demo_samples/test1.wav", 
                                      frame_ms=30, 
                                      hop_ms=10, 
                                      energy_threshold=0.01)
    print(f"提取了 {len(pitch_contour)} 帧基频: {pitch_contour[:10]}...")
"""

def get_pitch_contour(audio_path, sr=8000, frame_samples=256, hop_samples=256, window='hamming', energy_threshold='auto', apply_median=True):
    """
    读取整个音频文件，提取基频轨迹（序列）

    :param audio_path: 音频文件路径 (如 .wav)
    :param sr: 采样率，默认 8000 匹配 MIR-QBSH 数据集
    :param frame_samples: 帧长 (采样点数)，默认 256
    :param hop_samples: 帧移 (采样点数)，默认 256
    :param window: 传给单帧提取算法的窗函数类型
    :param apply_median: 是否应用中值滤波去噪，默认 True。
    :param energy_threshold: 能量阈值。'auto' 时自适应计算 (median*0.4, 最低 0.015)；
                             传入数值则使用固定阈值。低于阈值的帧视为静音，基频置 0。
    :return: 包含所有帧基频的一维 numpy 数组序列 (Pitch Contour)
    """
    # 1. 读取音频波形 (必须以相同采样率读取)
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    # 1.5 高通滤波去除低频环境噪声 (60Hz 以下人声基频极少出现)
    sos = butter(4, 60, btype='highpass', fs=sr, output='sos')
    y = sosfiltfilt(sos, y)

    # 2. 计算短时能量 (用于过滤静音)
    # 取均方根能量
    rms_energy = librosa.feature.rms(y=y, frame_length=frame_samples, hop_length=hop_samples, center=False)[0]

    # 2.5 自适应阈值：取非零帧 RMS 中位数的 40%，下限 0.015
    if energy_threshold == 'auto':
        rms_nonzero = rms_energy[rms_energy > 0]
        energy_threshold = max(np.median(rms_nonzero) * 0.4, 0.015) if len(rms_nonzero) > 0 else 0.015
    
    # 3. 执行分帧操作
    frames = librosa.util.frame(y, frame_length=frame_samples, hop_length=hop_samples)
    frames = frames.T 
    
    pitch_contour = []
    
    # 5. 遍历每一帧提基频
    for i, frame in enumerate(frames):
        # 能量检测机制：如果当前帧的能量低于设定的阈值，视为静音或无声音段 (Unvoiced)
        if rms_energy[i] < energy_threshold:
            pitch_contour.append(0.0)
            continue
            
        # 调用核心算法求基频 (Hz)
        f0 = extract_pitch(frame, sr, window=window)
        
        # 将频率 (Hz) 转换为音高 (MIDI Note Number)
        # 例如 440Hz 会被转换为 69。0 代表静音。
        if f0 > 0:
            pitch = librosa.hz_to_midi(f0)
            pitch_contour.append(pitch)
        else:
            pitch_contour.append(0.0)
            
    # 6. 利用中值滤波进行平滑去噪（去除离群的误判突刺）
    # 只有非0(有效发声)部分才应该被滤波，否则休止符会被抹平
    pitch_array = np.array(pitch_contour)
    if apply_median:
        valid_idx = pitch_array > 0
        if np.any(valid_idx):
            pitch_array[valid_idx] = medfilt(pitch_array[valid_idx], kernel_size=11)
        
    return pitch_array