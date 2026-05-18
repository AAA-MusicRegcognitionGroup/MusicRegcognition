import os
import numpy as np
import librosa
from process_audio import get_pitch_contour


def load_pv_file(pv_path):
    with open(pv_path, 'r') as f:
        lines = f.readlines()
    pv_seq = []
    for line in lines:
        val = float(line.strip())
        if val == 0:
            pv_seq.append(0.0)
        else:
            if val > 120:
                pv_seq.append(librosa.hz_to_midi(val))
            else:
                pv_seq.append(val)
    return np.array(pv_seq)


def evaluate_one(num_str, task_dir="demo_samples/task2"):
    wav_path = os.path.join(task_dir, f"{num_str}.wav")
    pv_path = os.path.join(task_dir, f"{num_str}.pv")

    if not os.path.exists(wav_path) or not os.path.exists(pv_path):
        return None

    extracted = get_pitch_contour(wav_path, sr=8000, frame_samples=256, hop_samples=256)
    pv = load_pv_file(pv_path)

    min_len = min(len(extracted), len(pv))
    ext = extracted[:min_len]
    pv_ref = pv[:min_len]

    valid = (pv_ref > 0)
    total_valid = int(np.sum(valid))
    if total_valid == 0:
        return None

    diff_abs = np.abs(ext - pv_ref)
    strict_correct = int(np.sum((diff_abs <= 1.0) & valid))
    mod_diff = np.abs((ext - pv_ref) % 12)
    octave_correct = int(np.sum(((mod_diff <= 1.0) | (mod_diff >= 11.0)) & (ext > 0) & valid))

    return {
        "strict_acc": (strict_correct / total_valid) * 100,
        "octave_acc": (octave_correct / total_valid) * 100,
        "mean_error": float(np.mean(mod_diff[valid])),
        "total_frames": total_valid,
    }


def main():
    print("=" * 70)
    print("  批量基频提取准确率评测 (00001 ~ 00014)")
    print("=" * 70)
    print(f"  {'编号':<8} {'有效帧数':<10} {'严格准确率':<12} {'八度容错':<12} {'平均半音误差':<14}")
    print("-" * 70)

    strict_list, octave_list, error_list = [], [], []
    missing = []

    for num in range(1, 15):
        num_str = str(num).zfill(5)
        result = evaluate_one(num_str)

        if result is None:
            missing.append(num_str)
            print(f"  {num_str:<8} {'（文件缺失或无效）':<46}")
            continue

        strict_list.append(result["strict_acc"])
        octave_list.append(result["octave_acc"])
        error_list.append(result["mean_error"])

        print(f"  {num_str:<8} {result['total_frames']:<10} "
              f"{result['strict_acc']:<12.2f}%{result['octave_acc']:<11.2f}%"
              f"{result['mean_error']:<14.3f}")

    print("-" * 70)

    if strict_list:
        print(f"  {'平均':<8} {'':<10} "
              f"{np.mean(strict_list):<12.2f}%{np.mean(octave_list):<11.2f}%"
              f"{np.mean(error_list):<14.3f}")
        print(f"  {'最高':<8} {'':<10} "
              f"{np.max(strict_list):<12.2f}%{np.max(octave_list):<11.2f}%"
              f"{np.min(error_list):<14.3f}")
        print(f"  {'最低':<8} {'':<10} "
              f"{np.min(strict_list):<12.2f}%{np.min(octave_list):<11.2f}%"
              f"{np.max(error_list):<14.3f}")

    if missing:
        print(f"\n  缺失/无效样本: {', '.join(missing)}")
    print(f"\n  共评测 {len(strict_list)} 份有效样本。")


if __name__ == "__main__":
    main()
