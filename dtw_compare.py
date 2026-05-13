import numpy as np
import librosa

# ==========================================
# 调用说明与使用方法
# ==========================================
# 
# 该模块实现了“基础2”的核心要求：使用 DTW（动态时间规整）算法对比两个音高序列的相似度。
# 包含零均值归一化处理，解决哼唱整体跑调（即使偏移好几个半音或几个八度，由于相对音高走势一致，相似度依然很高即可匹配）。
# 
# 示例用法：
"""
    from process_audio import get_pitch_contour
    from dtw_compare import calculate_similarity
    
    # 提取测试者的哼唱音高
    humming_seq = get_pitch_contour("demo_samples/humming.wav")
    
    # 这里为了演示，我们假设有了两首标准库中的旋律序列
    # (真实场景中可以通过读取 MIDI 文件转换得到)
    std_seq_1 = np.array([60, 62, 64, 60, 60, 62, 64, 60]) # 两只老虎
    std_seq_2 = np.array([67, 67, 64, 64, 62, 62, 60, 60]) # 其他旋律
    
    score1 = calculate_similarity(humming_seq, std_seq_1)
    score2 = calculate_similarity(humming_seq, std_seq_2)
    
    print(f"与歌曲1距离: {score1}, 与歌曲2距离: {score2}")
"""

def calculate_similarity(seq1, seq2, remove_silence=True):
    """
    通过 DTW 算法计算两个音高序列的距离（距离越小，相似度越高）。
    包含零均值归一化处理（Zero-Mean Normalization）。
    
    :param seq1: 第一个音高序列 (通常为哼唱提取的 pitch contour)
    :param seq2: 第二个音高序列 (通常为标准 MIDI 乐谱提取的 contour)
    :param remove_silence: 是否去除前后的静音段(值为0的帧)，默认去除。
    :return: 归一化后的 DTW 距离值。返回浮点数，数值越小越匹配。
    """
    seq1 = np.array(seq1, dtype=float)
    seq2 = np.array(seq2, dtype=float)
    
    # 1. 剔除静音帧 (值为 0 的帧)
    # 因为静音帧不参与旋律走势的比对，且如果带着 0 直接去做均值归一化，会严重拉低均值的准确性
    if remove_silence:
        seq1 = seq1[seq1 > 0]
        seq2 = seq2[seq2 > 0]
        
    # 如果剔除静音后序列太空，无法比较，直接返回无穷大距离
    if len(seq1) == 0 or len(seq2) == 0:
        return float('inf')
        
    # 2. 零均值归一化 (Zero-Mean Normalization)
    # 将两个序列减去各自的平均音高。
    # 作用：只要“音高的相对起伏”一致（例如 60-62-64 和 62-64-66），
    # 减去各自均值后它们就会变成一样的序列。完美解决男生女生八度不同、或者起调偏高偏低的问题。
    seq1_norm = seq1 - np.mean(seq1)
    seq2_norm = seq2 - np.mean(seq2)
    
    # librosa.sequence.dtw 要求输入维度为 (特征维度, 时间步数)
    # 因此我们需要使用 reshape(1, -1) 将 1D 数组变维
    X = seq1_norm.reshape(1, -1)
    Y = seq2_norm.reshape(1, -1)
    
    # 3. 计算 DTW 距离
    # metric='euclidean' 表示每对音高节点之间使用欧氏距离（绝对差值）计算路损
    D, wp = librosa.sequence.dtw(X, Y, metric='euclidean')
    
    # D[-1, -1] 存储的是到达终点的“累计最小代价 (Accumulated Cost)”
    total_cost = D[-1, -1]
    
    # 4. 路径长度归一化
    # 哼唱的快慢/长度不同会导致累加的代价不公平（较长的序列累计误差天然就大）。
    # 所以我们将总距离除以匹配路径的总步数 (len(wp))，得到平均每步的偏差。
    normalized_distance = total_cost / len(wp)
    
    return normalized_distance
