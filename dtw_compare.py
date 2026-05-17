import numpy as np
import librosa
import scipy.signal  # 用于二次滤波，消除残余突刺

# ==========================================
# 调用说明与使用方法
# ==========================================
# 该模块实现了针对“子序列匹配”和“现场录音”高度优化的相似度比对逻辑：
# 1. 二次中值滤波：彻底抹平自相关法产生的倍频/半频“突刺”。
# 2. 哼唱静音截断：只对用户哼唱进行静音截断，保留 MIDI 完整的时间轴脉络。
# 3. 音符量化：将浮点音高归类到整数半音，忽略嗓音自然抖动产生的细微偏差。
# 4. 改进型零均值归一化：计算 MIDI 均值时自动剔除休止符（0值），避免均值被稀释。
# 5. 子序列 DTW：允许片段匹配，无视录音前后的空白，自动在长乐谱中滑动寻找最优匹配。

def calculate_similarity(seq1, seq2, remove_silence=True):
    """
    通过改进的子序列 DTW 算法计算两个音高序列的距离。
    
    :param seq1: 哼唱音高序列 (短查询序列)
    :param seq2: 标准 MIDI 序列 (长基准序列)
    :param remove_silence: 是否去除哼唱前后的静音段，默认去除。
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
        # 【核心变更】：绝对不能裁剪 seq2 的 0！必须保留完整 MIDI 的时间轴结构
        # seq2 = seq2[seq2 > 0] 
        
    if len(seq1) < 5 or len(seq2) < 5:
        return float('inf')
        
    # 3. 音符量化 (Note Quantization)
    # 将浮点音高取整为标准半音编号，忽略 0.5 半音左右的自然嗓音抖动
    seq1 = np.round(seq1)
    seq2 = np.round(seq2)
    
    # 4. 改进型零均值归一化 (Zero-Mean Normalization)
    mean1 = np.mean(seq1)
    
    # 【核心变更】：计算 MIDI 均值时，只统计大于 0 的发声帧，防止被前奏或休止符的 0 稀释值
    midi_voice_frames = seq2[seq2 > 0]
    if len(midi_voice_frames) == 0:
        return float('inf')
    mean2 = np.mean(midi_voice_frames)
    
    seq1_norm = seq1 - mean1
    # 减去正确均值后，原曲中原本是 0 的休止符帧会变成一个很大的负数
    # 这个负数在 DTW 计算时会产生极高的惩罚代价，完美阻止系统将你的哼唱误匹配到空白段上
    seq2_norm = seq2 - mean2
    
    # 准备 DTW 输入数据 (调整为 librosa 要求的维数)
    X = seq1_norm.reshape(1, -1)
    Y = seq2_norm.reshape(1, -1)
    
    # 5. 子序列 DTW 核心计算
    # 设置 subseq=True，开启滑窗局部匹配模式
    D, wp = librosa.sequence.dtw(X, Y, metric='euclidean', subseq=True)
    
    # 在子序列匹配模式下，最终的最小累计代价位于累计代价矩阵 D 最后一行的最小值中
    total_cost = np.min(D[-1, :])
    
    # 6. 路径长度归一化
    # 消除哼唱长短对累加总代价的影响，计算平均每帧的偏差距离
    normalized_distance = total_cost / len(wp)
    
    return normalized_distance