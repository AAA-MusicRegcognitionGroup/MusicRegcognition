import os
import io
import base64
import glob
import uuid
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import pretty_midi

from flask import Flask, render_template, request, jsonify, Response
from scipy.io import wavfile

from process_audio import get_pitch_contour
from dtw_compare import calculate_similarity

app = Flask(__name__)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


def fig_to_base64():
    """Convert current matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close()
    return b64


def evaluate_one(num_str, task_dir="demo_samples/task2"):
    """Identical to batch_evaluate.evaluate_one."""
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

    # DTW distance between extracted pitch and MIDI reference
    midi_path = f"midi_cache/{num_str}.npy"
    if os.path.exists(midi_path):
        midi_seq = np.load(midi_path)
        dtw_dist = float(calculate_similarity(ext, midi_seq))
    else:
        dtw_dist = None

    return {
        "id": num_str,
        "strict_acc": round((strict_correct / total_valid) * 100, 2),
        "dtw_dist": round(dtw_dist, 4) if dtw_dist is not None else None,
        "mean_error": round(float(np.mean(mod_diff[valid])), 3),
        "total_frames": total_valid,
    }


def build_eval_chart(num_str):
    """Generate comparison chart for a single song evaluation."""
    wav_path = f"demo_samples/task2/{num_str}.wav"
    pv_path = f"demo_samples/task2/{num_str}.pv"
    if not os.path.exists(wav_path) or not os.path.exists(pv_path):
        return None

    extracted = get_pitch_contour(wav_path, sr=8000, frame_samples=256, hop_samples=256)
    pv_seq = load_pv_file(pv_path)
    min_len = min(len(extracted), len(pv_seq))
    ext = extracted[:min_len]
    pv_ref = pv_seq[:min_len]

    plot_ext = np.where(ext > 0, ext, np.nan)
    plot_pv = np.where(pv_ref > 0, pv_ref, np.nan)

    plt.figure(figsize=(10, 3.8), dpi=100)
    plt.plot(plot_ext, label='算法提取', color='#1A5276', linewidth=1.2)
    plt.plot(plot_pv, label='人工标注 (.pv)', color='#CD6155', linestyle='--', linewidth=1.8)
    plt.title(f'歌曲 {num_str}  基频提取准确度对比', fontsize=12, fontweight='bold')
    plt.xlabel('帧序号')
    plt.ylabel('MIDI 音高')
    plt.grid(True, linestyle=':', alpha=0.5)
    valid_p = pv_ref[pv_ref > 0]
    if len(valid_p) > 0:
        plt.ylim([np.min(valid_p) - 2, np.max(valid_p) + 2])
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    return fig_to_base64()


def build_match_chart(live_seq, midi_seq, song_name):
    """Generate live-vs-MIDI comparison chart."""
    t_live = np.arange(len(live_seq)) * 32 / 1000
    t_midi = np.arange(len(midi_seq)) * 32 / 1000
    plot_live = np.where(live_seq > 0, live_seq, np.nan)
    plot_midi = np.where(midi_seq > 0, midi_seq, np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.8), dpi=100,
                                    gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(t_live, plot_live, color='#1A5276', linewidth=1.0, label='实时录音')
    ax1.plot(t_midi, plot_midi, color='#CD6155', linestyle='--', linewidth=1.5,
             alpha=0.8, label=f'{song_name}（标准乐谱）')
    all_v = np.concatenate([live_seq[live_seq > 0], midi_seq[midi_seq > 0]])
    if len(all_v) > 0:
        ax1.set_ylim([np.min(all_v) - 3, np.max(all_v) + 3])
    ax1.set_title(f'实时录音 vs {song_name}（完整时间轴）', fontsize=11, fontweight='bold')
    ax1.set_ylabel('MIDI 音高')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, linestyle=':', alpha=0.5)

    max_t = min(len(live_seq), len(midi_seq)) * 32 / 1000
    zoom_end = min(5, max_t)
    ax2.plot(t_live, plot_live, color='#1A5276', linewidth=1.0)
    ax2.plot(t_midi, plot_midi, color='#CD6155', linestyle='--', linewidth=1.5, alpha=0.8)
    if len(all_v) > 0:
        ax2.set_ylim([np.min(all_v) - 3, np.max(all_v) + 3])
    ax2.set_xlim([0, zoom_end])
    ax2.set_title('前 5 秒放大', fontsize=10)
    ax2.set_xlabel('时间（秒）')
    ax2.set_ylabel('MIDI 音高')
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    return fig_to_base64()


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/batch_evaluate', methods=['POST'])
def batch_evaluate():
    results = []
    missing = []
    for num in range(1, 15):
        num_str = str(num).zfill(5)
        r = evaluate_one(num_str)
        if r is None:
            missing.append(num_str)
        else:
            results.append(r)

    strict_vals = [r['strict_acc'] for r in results]
    dtw_vals = [r['dtw_dist'] for r in results if r['dtw_dist'] is not None]
    error_vals = [r['mean_error'] for r in results]

    summary = {}
    if results:
        summary = {
            "avg_strict": round(np.mean(strict_vals), 2),
            "avg_dtw": round(np.mean(dtw_vals), 4) if dtw_vals else None,
            "avg_error": round(np.mean(error_vals), 3),
            "max_strict": round(np.max(strict_vals), 2),
            "min_dtw": round(np.min(dtw_vals), 4) if dtw_vals else None,
            "max_dtw": round(np.max(dtw_vals), 4) if dtw_vals else None,
            "best_error": round(np.min(error_vals), 3),
            "min_strict": round(np.min(strict_vals), 2),
            "worst_error": round(np.max(error_vals), 3),
            "total_evaluated": len(results),
        }

    return jsonify({
        "results": results,
        "missing": missing,
        "summary": summary,
    })


@app.route('/api/evaluate/<song_id>', methods=['POST'])
def evaluate_single(song_id):
    num_str = song_id.zfill(5)
    r = evaluate_one(num_str)
    if r is None:
        return jsonify({"error": f"歌曲 {num_str} 不存在或无效"}), 404
    chart_b64 = build_eval_chart(num_str)
    r["chart"] = chart_b64

    # DTW ranking: compare extracted pitch against ALL MIDI files
    wav_path = f"demo_samples/task2/{num_str}.wav"
    extracted = get_pitch_contour(wav_path, sr=8000, frame_samples=256, hop_samples=256)
    cache_dir = "midi_cache/"
    npy_files = sorted(glob.glob(os.path.join(cache_dir, "*.npy")))
    ranking = []
    for n_file in npy_files:
        mid = os.path.basename(n_file).replace(".npy", "")
        midi_seq = np.load(n_file)
        score = calculate_similarity(extracted, midi_seq)
        ranking.append({"song": mid, "dist": round(float(score), 4)})
    ranking.sort(key=lambda x: x["dist"])
    r["ranking"] = ranking
    r["dtw_rank"] = next((i + 1 for i, item in enumerate(ranking) if item["song"] == num_str), None)

    return jsonify(r)


@app.route('/api/match', methods=['POST'])
def match_live():
    if 'audio' not in request.files:
        return jsonify({"error": "未上传音频文件"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "文件名为空"}), 400

    upload_dir = "demo_samples/进阶1"
    os.makedirs(upload_dir, exist_ok=True)

    # Save raw upload (any format)
    ext = os.path.splitext(file.filename)[1] or ".tmp"
    raw_path = os.path.join(upload_dir, f"raw_{uuid.uuid4().hex[:8]}{ext}")
    file.save(raw_path)

    # Convert to standardized 8000Hz mono 16-bit WAV (handles webm/wav/anything)
    y, _ = librosa.load(raw_path, sr=8000, mono=True)
    uid = uuid.uuid4().hex[:8]
    save_path = os.path.join(upload_dir, f"upload_{uid}.wav")
    wavfile.write(save_path, 8000, (y * 32767).astype(np.int16))

    # Voice separation (optional, controlled by frontend)
    separate = request.form.get('separate', 'true').lower() == 'true'
    vocal_path = save_path
    separation_used = False
    if separate:
        try:
            from split_demo import run_audio_separation
            output_files = run_audio_separation(save_path, output_dir=upload_dir)
            for f in output_files:
                basename = os.path.basename(f)
                if "(Vocals)" in basename or "vocals" in basename.lower():
                    if os.path.exists(f):
                        vocal_path = f
                        separation_used = True
                        break
        except Exception:
            pass

    # Extract pitch contour
    live_seq = get_pitch_contour(vocal_path, sr=8000, frame_samples=256, hop_samples=256)

    # Match against MIDI cache
    cache_dir = "midi_cache/"
    npy_files = sorted(glob.glob(os.path.join(cache_dir, "*.npy")))
    if not npy_files:
        return jsonify({"error": "未找到 MIDI 缓存，请先运行 process_midi.py。"}), 500

    results = []
    for n_file in npy_files:
        song_id = os.path.basename(n_file).replace(".npy", "")
        midi_seq = np.load(n_file)
        score = calculate_similarity(live_seq, midi_seq)
        results.append({"song": song_id, "dist": round(float(score), 4)})

    results.sort(key=lambda x: x["dist"])

    # Build chart for top match
    top_song = results[0]["song"]
    top_midi = np.load(os.path.join(cache_dir, f"{top_song}.npy"))
    chart_b64 = build_match_chart(live_seq, top_midi, top_song)

    return jsonify({
        "results": results[:10],
        "top_match": top_song,
        "chart": chart_b64,
        "separation_used": separation_used,
    })


@app.route('/api/songs', methods=['GET'])
def list_songs():
    midi_dir = "demo_samples/midiFile"
    songs = []
    if os.path.isdir(midi_dir):
        for f in sorted(os.listdir(midi_dir)):
            if f.endswith('.mid'):
                songs.append(os.path.splitext(f)[0])
    return jsonify(songs)


def synthesize_midi_to_wav(song_id):
    """Synthesize MIDI to WAV bytes in memory, with on-disk caching."""
    cache_dir = "midi_audio_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{song_id}.wav")

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return f.read()

    midi_path = f"demo_samples/midiFile/{song_id}.mid"
    if not os.path.exists(midi_path):
        return None

    midi = pretty_midi.PrettyMIDI(midi_path)
    # Use built-in sine-wave synthesis — no SoundFont needed
    audio = midi.synthesize(fs=22050)
    audio_int16 = (audio * 32767 * 0.8).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, 22050, audio_int16)
    wav_bytes = buf.getvalue()

    with open(cache_path, 'wb') as f:
        f.write(wav_bytes)

    return wav_bytes


@app.route('/api/midi_audio/<song_id>', methods=['GET'])
def midi_audio(song_id):
    wav_bytes = synthesize_midi_to_wav(song_id)
    if wav_bytes is None:
        return jsonify({"error": f"MIDI {song_id} 不存在"}), 404
    return Response(wav_bytes, mimetype='audio/wav')


if __name__ == '__main__':
    os.makedirs("demo_samples/进阶1", exist_ok=True)
    print("=" * 50)
    print("  哼唱识曲 Web 界面")
    print("  浏览器打开 http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
