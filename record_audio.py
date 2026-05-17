import os
import wave
import pyaudio
import numpy as np # 利用 numpy 进行数值归一化

def record_audio_to_project(file_name="live_test.wav", duration=5):
    """
    录制音频、进行音量归一化，并保存到项目的 demo_samples/进阶1/ 目录下
    """
    save_dir = "demo_samples/进阶1/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    output_path = os.path.join(save_dir, file_name)

    # 配置参数
    SR = 8000 
    CHUNK = 1024
    FORMAT = pyaudio.paInt16 # 16位深度，数值范围 -32768 到 32767
    CHANNELS = 1

    p = pyaudio.PyAudio()

    print(f">>> 准备录音 (时长: {duration}秒)...")
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=SR,
                    input=True, frames_per_buffer=CHUNK)

    print(">>> 正在录制，请开始哼唱...")
    frames = []

    for _ in range(0, int(SR / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print(">>> 录音结束，正在进行音量归一化处理...")

    # --- 音量归一化处理逻辑 ---
    # 1. 将字节流转换为 np.int16 数组
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    
    # 2. 找到当前序列的最大绝对值
    max_val = np.max(np.abs(audio_data))
    
    if max_val > 0:
        # 3. 将振幅缩放到 16-bit 的最大范围 (32767)
        # 使用 float64 进行中间计算防止溢出，最后转回 int16
        normalized_data = (audio_data.astype(np.float64) / max_val) * 32767.0
        audio_bytes = normalized_data.astype(np.int16).tobytes()
    else:
        audio_bytes = b''.join(frames) # 如果全是静音，保持原样

    # 停止并关闭流
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 写入 WAV 文件
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SR)
        wf.writeframes(audio_bytes)

    print(f">>> 归一化后的文件已保存至: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    record_audio_to_project(file_name="live_test.wav", duration=10)