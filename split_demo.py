# Windows: 使用 scoop install ffmpeg 或去官网下载并配置系统环境变量。
# Mac: brew install ffmpeg
# Linux: sudo apt install ffmpeg

# 如果你有 NVIDIA 显卡（推荐，速度极快）
# pip install "audio-separator[gpu]"
# 如果只有 CPU 运行
# pip install "audio-separator[cpu]"


import os
import logging
from audio_separator.separator import Separator

def select_best_available_model(separator):
    """
    动态从当前库支持的列表中筛选最佳模型，彻底解决版本名字对不上的问题
    """
    # 获取当前版本支持的所有模型文件名列表
    try:
        supported_models = list(separator.model_data.keys())
    except AttributeError:
        # 万一获取失败，使用最稳妥的保底模型名
        return "Kim_Vocal_2.onnx"

    if not supported_models:
        raise RuntimeError("无法获取支持的模型列表，请检查网络或库安装是否完整。")

    print("\n🔍 正在从你本地的库中自动匹配最佳 AI 模型...")
    
    # 策略 1：优先找 RoFormer 模型（现代音质天花板）
    for model in supported_models:
        if "roformer" in model.lower() and model.endswith(".onnx"):
            print(f"✨ 成功自动匹配到 RoFormer 模型: {model}")
            return model

    # 策略 2：次选找 Kim 或者是高质 MDX-Net 人声模型
    for model in supported_models:
        if "kim" in model.lower() or "vocal_2" in model.lower():
            print(f"✨ 成功自动匹配到经典人声模型: {model}")
            return model

    # 策略 3：实在没有，拿列表里的第一个模型保底
    print(f"⚠️ 未找到指定推荐模型，自动启用兼容保底模型: {supported_models[0]}")
    return supported_models[0]


def run_audio_separation(input_file_path, output_dir="./output_stems"):
    """
    动态匹配的人声伴奏分离核心函数
    """
    # 确保输出和模型缓存目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_dir = "./separator_models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("=" * 60)
    print(f"🎵 初始化分离器... 输出目录: {output_dir}")
    print("=" * 60)

    # 1. 初始化 Separator
    separator = Separator(
        output_dir=output_dir,
        model_file_dir=model_dir,  # 确保 Windows 写入权限
        output_format="WAV",       # 可改为 WAV、FLAC
        log_level=logging.WARNING   # 减少冗余日志，只看核心输出
    )

    # 2. 动态获取并加载模型（核心修复：绝不硬编码名字）
    model_filename = select_best_available_model(separator)
    
    print(f"\n[1/2] 正在加载/下载 AI 模型: {model_filename} ...")
    print("提示：如果是首次运行该模型，会自动下载（约几百MB），请稍等。")
    separator.load_model(model_filename)

    # 3. 执行音频分离
    print(f"\n[2/2] 开始分离音频文件: {input_file_path} ...")
    print("提示：当前为 CPU 模式，分离一首歌大约需要 1~3 分钟，请耐心等待...")
    
    output_files = separator.separate(input_file_path)

    print("\n" + "=" * 60)
    print("🎉 音频分离成功完成！")
    print("=" * 60)
    full_paths = []
    for file_name in output_files:
        full_path = os.path.join(output_dir, file_name)
        full_paths.append(full_path)
        print(f"👉 已成功保存: {full_path}")

    return full_paths


if __name__ == "__main__":
    # 目标音频文件名
    target_audio = "AI_demo.wav" 
    
    if os.path.exists(target_audio):
        run_audio_separation(target_audio)
    else:
        print(f"❌ 错误：未在当前目录下找到测试音频 '{target_audio}'")
        print(f"请确保该文件存在于 D:\\Work\\MusicRegcognition\\ 文件夹下再运行！")