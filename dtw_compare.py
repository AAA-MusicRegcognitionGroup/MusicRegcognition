import numpy as np
import librosa
import scipy.signal  # 用于二次滤波，消除残余突刺

# ==========================================
# 调用说明与使用方法
# ==========================================
# 该模块实现了针对”子序列匹配”和”现场录音”高度优化的相似度比对逻辑：
# 1. 二次中值滤波：彻底抹平自相关法产生的倍频/半频”突刺”。
# 2. 差异化静音处理：只剔除哼唱的 0，保留 MIDI 完整时间轴。
# 3. 音符量化：将浮点音高归类到整数半音，忽略嗓音自然抖动产生的细微偏差。
# 4. 滑动窗局部零均值归一化：每个 MIDI 帧减去自身周围窗口内发声帧的均值，
#    消除长 MIDI 全局均值对局部匹配段的偏置，同时保持子序列 DTW 的搜索能力。
# 5. 子序列 DTW：允许片段匹配，自动在长乐谱中滑动寻找最优匹配。

def calculate_similarity(seq1, seq2, remove_silence=True, local_window=350):
    """
    通过滑动窗局部归一化 + 子序列 DTW 计算两个音高序列的距离。

    :param seq1: 哼唱音高序列 (短查询序列)
    :param seq2: 标准 MIDI 序列 (长基准序列)
    :param remove_silence: 是否去除哼唱前后的静音段，默认去除。
    :param local_window: MIDI 局部归一化的窗口大小 (帧)，默认 350 (11.2s@32ms)。
                         窗口越大越接近全局归一化，越小越局部化。
    :return: 归一化后的 DTW 距离。数值越小越匹配。
    """
    seq1 = np.array(seq1, dtype=float)
    seq2 = np.array(seq2, dtype=float)

    # 1. 二次中值滤波 (针对用户录音中的倍频/半频突刺进行强力抹平)
    if len(seq1) > 7:
        seq1 = scipy.signal.medfilt(seq1, kernel_size=7)

    # 2. 差异化静音处理
    if remove_silence:
        seq1 = seq1[seq1 > 0]  # 哼唱片段必须剔除静音，避免干扰
        # 保留完整 MIDI 时间轴结构

    if len(seq1) < 5 or len(seq2) < 5:
        return float('inf')

    # 3. 音符量化 (Note Quantization)
    seq1 = np.round(seq1)
    seq2 = np.round(seq2)

    # 4. 归一化
    # seq1: 全局零均值 (短序列，全局均值 = 局部均值)
    mean1 = np.mean(seq1)
    seq1_norm = seq1 - mean1

    # seq2: 滑动窗局部零均值 (消除长 MIDI 全局均值对局部匹配段的偏置)
    half = local_window // 2
    seq2_norm = np.zeros_like(seq2)
    for i in range(len(seq2)):
        if seq2[i] == 0:
            seq2_norm[i] = 0
            continue
        lo = max(0, i - half)
        hi = min(len(seq2), i + half)
        local_voiced = seq2[lo:hi][seq2[lo:hi] > 0]
        if len(local_voiced) > 0:
            seq2_norm[i] = seq2[i] - np.mean(local_voiced)
        # else: 保持 0

    # 5. 子序列 DTW 核心计算
    X = seq1_norm.reshape(1, -1)
    Y = seq2_norm.reshape(1, -1)
    D, wp = librosa.sequence.dtw(X, Y, metric='euclidean', subseq=True)

    total_cost = np.min(D[-1, :])
    normalized_distance = total_cost / len(wp)

    return normalized_distance