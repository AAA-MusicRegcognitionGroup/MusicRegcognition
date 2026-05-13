# 自相关法提取简单单音音频的基频音高序列
# ==========================================
# 调用说明与使用方法
# ==========================================
# 1. 该函数设计用于处理【单帧音频】，而不是完整的音频文件。
#    在实际工程中，你需要先将音频使用移动窗口（如窗长30ms，步长10ms）分帧，
#    然后对每一帧循环调用该函数，从而得到一条基频轨迹（Pitch Contour）。
#
# 2. 示例用法：
'''
    import librosa
    y, sr = librosa.load('sample.wav', sr=None)
    frame = y[0:int(0.03*sr)]          # 截取第一帧 (30ms)

    # 使用默认的汉明窗
    pitch = extract_pitch(frame, sr)

    # 或者指定使用布莱克曼窗 (blackman) / 不加窗 (None)
    pitch = extract_pitch(frame, sr, window='blackman')
    print(f"帧基频: {pitch} Hz")

'''

import numpy as np

def extract_pitch(frame, fs, window='hamming'):
    """
    使用自相关法提取一帧音频的基频
    :param frame: 一维 numpy 数组，代表一帧音频波形
    :param fs: 采样率 (Hz)
    :param window: 窗函数类型，支持 'hamming', 'blackman', 或 None(不加窗)
    :return: 估计的基频 (Hz)，如果没找到有效基频则返回 0
    """
    # 1. 窗函数处理 (减少边缘效应，抑制频谱泄漏)
    if window == 'hamming':
        frame = frame * np.hamming(len(frame))
    elif window == 'blackman':
        frame = frame * np.blackman(len(frame))
    elif window is not None:
        raise ValueError("不支持的窗类型。请选择 'hamming', 'blackman' 或 None。")

    # 2. 计算自相关
    # np.correlate 默认计算完整的交叉相关，返回长度为 2N-1
    # 我们只需要正延迟部分（后半段）
    corr = np.correlate(frame, frame, mode='full')
    corr = corr[len(corr)//2:]
    
    # 3. 限定合理的搜索范围，过滤干扰
    # 人类哼唱的频率范围通常在 50Hz (极低音) 到 1000Hz (极高音) 之间
    min_freq = 50
    max_freq = 1000
    
    # 将频率范围转换为延迟点数范围 (tau)
    min_lag = int(fs / max_freq)
    max_lag = int(fs / min_freq)
    
    # 确保 max_lag 不超过帧长
    max_lag = min(max_lag, len(corr))
    
    # 4. 截取有效范围内的自相关结果
    valid_corr = corr[min_lag:max_lag]
    
    if len(valid_corr) == 0:
        return 0
        
    # 5. 找到有效范围内的最大峰值索引
    # 注意：这个索引是从 min_lag 开始算的，需要加回去
    peak_idx = np.argmax(valid_corr)
    tau_peak = peak_idx + min_lag
    
    # 6. 计算基频
    f0 = fs / tau_peak
    
    return f0