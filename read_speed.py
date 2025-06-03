import time
import random
import os
import itertools # For iterating over durations and sample rates
import math

# Try to import all libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Librosa not installed, skipping librosa tests.")

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("Torchaudio not installed, skipping torchaudio tests.")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Soundfile not installed, skipping soundfile tests.")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Pydub not installed, skipping pydub tests. Note: Pydub requires ffmpeg or libav for non-wav files.")

try:
    import wave
    import numpy as np # wave library reads bytes, numpy is often used for conversion
    WAVE_AVAILABLE = True
except ImportError:
    WAVE_AVAILABLE = False
    print("Wave (standard library) or numpy not installed, skipping wave tests.")

try:
    import matplotlib
    matplotlib.use('Agg') # Use Agg backend for non-interactive plotting, especially in environments without a display
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not installed, skipping plot generation.")


# --- Configuration Parameters ---
DURATIONS_S = [5, 10, 20, 40, 60]  # Durations in seconds for full read tests
SAMPLE_RATES_HZ = [16000, 22050, 48000] # Sample rates in Hz

RANDOM_READ_TOTAL_DURATION_S = 300 # 5 minutes
RANDOM_READ_SR = 16000 # Sample rate for the long file used in random read test
RANDOM_READ_BASE_FILENAME = f"audio_{RANDOM_READ_TOTAL_DURATION_S}s_{RANDOM_READ_SR}Hz.wav"
RANDOM_SEGMENT_DURATION_S = 5  # Seconds
NUM_ITERATIONS = 100  # Number of times to read each file for averaging (adjust as needed)

# --- Helper Functions (same as before) ---
def get_file_duration_sf(filepath):
    try:
        f = sf.SoundFile(filepath)
        return len(f) / f.samplerate
    except Exception: return None

def generate_random_start_time_ms(total_duration_s, segment_duration_s):
    if total_duration_s is None or total_duration_s < segment_duration_s: return None
    max_start_s = total_duration_s - segment_duration_s
    random_start_s = random.uniform(0, max_start_s)
    return int(random_start_s * 1000)

# --- Test Functions (same as before, ensure they handle sr correctly) ---
def test_read_librosa(filepath, segment_start_ms=None, segment_duration_s=None):
    start_time = time.perf_counter()
    if segment_start_ms is not None and segment_duration_s is not None:
        offset = segment_start_ms / 1000
        duration = segment_duration_s
        try: y, sr = librosa.load(filepath, sr=None, offset=offset, duration=duration)
        except Exception as e: print(f"Librosa error (segment: {filepath}): {e}"); return float('inf')
    else:
        try: y, sr = librosa.load(filepath, sr=None)
        except Exception as e: print(f"Librosa error (full: {filepath}): {e}"); return float('inf')
    return time.perf_counter() - start_time

def test_read_torchaudio(filepath, segment_start_ms=None, segment_duration_s=None):
    start_time = time.perf_counter()
    if segment_start_ms is not None and segment_duration_s is not None:
        try:
            info = torchaudio.info(filepath)
            sr = info.sample_rate
            frame_offset = int((segment_start_ms / 1000) * sr)
            num_frames = int(segment_duration_s * sr)
            waveform, sample_rate = torchaudio.load(filepath, frame_offset=frame_offset, num_frames=num_frames)
        except Exception as e: print(f"Torchaudio error (segment: {filepath}): {e}"); return float('inf')
    else:
        try: waveform, sample_rate = torchaudio.load(filepath)
        except Exception as e: print(f"Torchaudio error (full: {filepath}): {e}"); return float('inf')
    return time.perf_counter() - start_time

def test_read_soundfile(filepath, segment_start_ms=None, segment_duration_s=None):
    start_time = time.perf_counter()
    if segment_start_ms is not None and segment_duration_s is not None:
        try:
            with sf.SoundFile(filepath, 'r') as f:
                sr = f.samplerate
                start_frame = int((segment_start_ms / 1000) * sr)
                num_frames = int(segment_duration_s * sr)
                f.seek(start_frame)
                data = f.read(frames=num_frames, dtype='float32')
        except Exception as e: print(f"Soundfile error (segment: {filepath}): {e}"); return float('inf')
    else:
        try: data, samplerate = sf.read(filepath)
        except Exception as e: print(f"Soundfile error (full: {filepath}): {e}"); return float('inf')
    return time.perf_counter() - start_time

def test_read_pydub(filepath, segment_start_ms=None, segment_duration_s=None):
    start_time = time.perf_counter()
    try:
        audio = AudioSegment.from_file(filepath)
        if segment_start_ms is not None and segment_duration_s is not None:
            end_ms = segment_start_ms + (segment_duration_s * 1000)
            segment = audio[segment_start_ms:end_ms]
    except Exception as e: print(f"Pydub error ({filepath}): {e}"); return float('inf')
    return time.perf_counter() - start_time

def test_read_wave(filepath, segment_start_ms=None, segment_duration_s=None):
    if not filepath.lower().endswith(".wav"): return float('nan')
    start_time = time.perf_counter()
    try:
        with wave.open(filepath, 'rb') as wf:
            framerate = wf.getframerate()
            if segment_start_ms is not None and segment_duration_s is not None:
                start_frame = int((segment_start_ms / 1000) * framerate)
                num_frames_to_read = int(segment_duration_s * framerate)
                wf.setpos(start_frame)
                frames = wf.readframes(num_frames_to_read)
            else:
                frames = wf.readframes(wf.getnframes())
    except Exception as e: print(f"Wave module error ({filepath}): {e}"); return float('inf')
    return time.perf_counter() - start_time


# --- Plotting Function ---
def generate_plots(all_results_data, libraries_config, output_filename="audio_library_read_speed_comparison.png"):
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot generate plots.")
        return
    if not all_results_data:
        print("No results data to plot.")
        return

    num_tests = len(all_results_data)
    if num_tests == 0:
        print("No test cases found in results data.")
        return

    # Determine grid size (prefer more rows if not a perfect square to keep width manageable)
    ncols = math.ceil(math.sqrt(num_tests))
    if num_tests > 4 : # For more plots, try to make it wider
        ncols = 4 if num_tests > 12 else 3 if num_tests > 6 else 2
    if num_tests <= 2: ncols = num_tests # Handle 1 or 2 plots in a single row
    nrows = math.ceil(num_tests / ncols)


    fig_width = ncols * 4.5  # Adjust cell width
    fig_height = nrows * 4   # Adjust cell height
    # Ensure a minimum figure size
    fig_width = max(fig_width, 8)
    fig_height = max(fig_height, 6)


    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten() # Flatten to 1D array for easy iteration

    library_names = [lib[0] for lib in libraries_config]
    x_pos = np.arange(len(library_names))

    for i, test_data in enumerate(all_results_data):
        if i >= len(axes): # Should not happen if nrows, ncols calculated correctly
            print(f"Warning: More test data ({len(all_results_data)}) than available subplots ({len(axes)}). Some plots will be skipped.")
            break
        ax = axes[i]
        test_case_name = test_data["Test Case"]
        
        times = []
        for lib_name in library_names:
            val = test_data.get(lib_name)
            if isinstance(val, (int, float)) and not np.isnan(val) and not np.isinf(val) and val is not None:
                times.append(float(val))
            else: # ERROR, N/A, MISSING_FILE, None
                times.append(np.nan) # Matplotlib typically skips plotting NaNs for bars

        bars = ax.bar(x_pos, times, align='center', alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']) # Default MPL colors
        ax.set_xticks(x_pos)
        ax.set_xticklabels(library_names, rotation=45, ha="right", fontsize=8)
        ax.set_title(test_case_name, fontsize=10, wrap=True)
        ax.set_ylabel("Time (s)", fontsize=8)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        # Add value labels on top of bars
        for bar_idx, bar_obj in enumerate(bars):
            yval = bar_obj.get_height()
            if not np.isnan(yval):
                ax.text(bar_obj.get_x() + bar_obj.get_width()/2.0, yval + (ax.get_ylim()[1] * 0.01) , f"{yval:.4f}", ha='center', va='bottom', fontsize=7, rotation=30)


    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle and bottom
    fig.suptitle("Audio Library Read Speed Comparison", fontsize=16, y=0.99)
    
    try:
        plt.savefig(output_filename, dpi=150) # Lower dpi for faster save and smaller file if 300 is too slow/large
        print(f"\nPlot saved as {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")


# --- Main Test Logic ---
def run_tests():
    all_results_data = [] 
    libraries_config = []
    if LIBROSA_AVAILABLE: libraries_config.append(("Librosa", test_read_librosa))
    if TORCHAUDIO_AVAILABLE: libraries_config.append(("Torchaudio", test_read_torchaudio))
    if SOUNDFILE_AVAILABLE: libraries_config.append(("Soundfile", test_read_soundfile))
    if PYDUB_AVAILABLE: libraries_config.append(("Pydub", test_read_pydub))
    if WAVE_AVAILABLE: libraries_config.append(("Wave", test_read_wave)) # Shorter name for plot

    if not libraries_config:
        print("No audio libraries available for testing.")
        return

    print("--- Test 1: Full Audio File Read Speed ---")
    for duration_s in DURATIONS_S:
        for sr_hz in SAMPLE_RATES_HZ:
            filename = f"audio_{duration_s}s_{sr_hz}Hz.wav"
            sr_khz_str = f"{sr_hz/1000:.0f}kHz" if sr_hz != 22050 else "22.05kHz"
            test_case_name = f"{duration_s}s @ {sr_khz_str}"

            if not os.path.exists(filename):
                print(f"File {filename} not found! Skipping.")
                result_row = {"Test Case": test_case_name}
                for lib_name, _ in libraries_config: result_row[lib_name] = "MISSING"
                all_results_data.append(result_row)
                continue

            print(f"\nTesting {filename} ({test_case_name})")
            current_results = {"Test Case": test_case_name}
            for lib_name, lib_func in libraries_config:
                if lib_name == "Wave" and not filename.lower().endswith(".wav"):
                    current_results[lib_name] = float('nan')
                    continue
                total_time = 0; error_occurred = False
                print(f"  Testing {lib_name}...")
                for _ in range(NUM_ITERATIONS):
                    time_taken = lib_func(filename)
                    if time_taken == float('inf'): error_occurred = True; break
                    total_time += time_taken
                if error_occurred: current_results[lib_name] = float('inf')
                else: current_results[lib_name] = total_time / NUM_ITERATIONS; print(f"    Avg: {current_results[lib_name]:.6f}s")
            all_results_data.append(current_results)

    print("\n--- Test 2: Random Segment Read Speed ---")
    sr_khz_str_rand = f"{RANDOM_READ_SR/1000:.0f}kHz" if RANDOM_READ_SR != 22050 else "22.05kHz"
    test_case_name_random = f"{RANDOM_READ_TOTAL_DURATION_S//60}min@{sr_khz_str_rand} (Rand {RANDOM_SEGMENT_DURATION_S}s)"
    
    if not os.path.exists(RANDOM_READ_BASE_FILENAME):
        print(f"File {RANDOM_READ_BASE_FILENAME} not found! Skipping random read.")
        result_row = {"Test Case": test_case_name_random}
        for lib_name, _ in libraries_config: result_row[lib_name] = "MISSING"
        all_results_data.append(result_row)
    else:
        print(f"\nTesting {RANDOM_READ_BASE_FILENAME} (Random {RANDOM_SEGMENT_DURATION_S}s segment)")
        current_results = {"Test Case": test_case_name_random}
        total_duration_file_s = get_file_duration_sf(RANDOM_READ_BASE_FILENAME) or \
                                (librosa.get_duration(filename=RANDOM_READ_BASE_FILENAME) if LIBROSA_AVAILABLE else None)

        if total_duration_file_s is None or total_duration_file_s < RANDOM_SEGMENT_DURATION_S:
            print(f"  Cannot run random test on {RANDOM_READ_BASE_FILENAME} (duration issue).")
            for lib_name, _ in libraries_config: current_results[lib_name] = "N/A"
        else:
            for lib_name, lib_func in libraries_config:
                if lib_name == "Wave" and not RANDOM_READ_BASE_FILENAME.lower().endswith(".wav"):
                    current_results[lib_name] = float('nan')
                    continue
                total_time = 0; error_occurred = False
                print(f"  Testing {lib_name}...")
                for _ in range(NUM_ITERATIONS):
                    random_start_ms = generate_random_start_time_ms(total_duration_file_s, RANDOM_SEGMENT_DURATION_S)
                    if random_start_ms is None: error_occurred = True; break
                    time_taken = lib_func(RANDOM_READ_BASE_FILENAME, random_start_ms, RANDOM_SEGMENT_DURATION_S)
                    if time_taken == float('inf'): error_occurred = True; break
                    total_time += time_taken
                if error_occurred: current_results[lib_name] = float('inf')
                else: current_results[lib_name] = total_time / NUM_ITERATIONS; print(f"    Avg: {current_results[lib_name]:.6f}s")
        all_results_data.append(current_results)
    
    # --- Generate Markdown Summary ---
    print("\n\n--- Summary of Results (Average Time in Seconds) ---")
    header_cols = ["Test Case"] + [lib[0] for lib in libraries_config]
    markdown_table = "| " + " | ".join(header_cols) + " |\n"
    markdown_table += "|-" + "-|-".join(["-" * len(h) for h in header_cols]) + "-|\n"
    for res_item in all_results_data:
        row_vals = [res_item["Test Case"]]
        for lib_name, _ in libraries_config:
            val = res_item.get(lib_name)
            if isinstance(val, str): row_vals.append(val) # MISSING, N/A
            elif val == float('inf'): row_vals.append("ERROR")
            elif val == float('nan') or val is None: row_vals.append("N/A")
            else: row_vals.append(f"{val:.6f}")
        markdown_table += "| " + " | ".join(row_vals) + " |\n"
    print("\nMarkdown Formatted Results:\n")
    print(markdown_table)

    # --- Generate Plot ---
    generate_plots(all_results_data, libraries_config)


if __name__ == "__main__":
    # --- Generate dummy audio files if they don't exist ---
    try:
        import numpy # Already imported usually, but good for explicitness here
        from scipy.io import wavfile

        def create_dummy_wav(filename, duration_s, sr=16000):
            if not os.path.exists(filename):
                print(f"Creating dummy: {filename} ({duration_s}s, {sr}Hz)...")
                amp = np.iinfo(np.int16).max / 3
                samples = int(duration_s * sr)
                freq = 440
                t = np.linspace(0, duration_s, samples, endpoint=False)
                audio_data = amp * np.sin(2 * np.pi * freq * t)
                wavfile.write(filename, sr, audio_data.astype(np.int16))

        print("Checking/Creating dummy audio files...")
        for dur in DURATIONS_S:
            for sr_val in SAMPLE_RATES_HZ:
                create_dummy_wav(f"audio_{dur}s_{sr_val}Hz.wav", dur, sr_val)
        create_dummy_wav(RANDOM_READ_BASE_FILENAME, RANDOM_READ_TOTAL_DURATION_S, RANDOM_READ_SR)
        print("Dummy file check complete.\n")

    except ImportError:
        print("Numpy or Scipy not installed. Cannot create dummy audio files. Please create them manually or install these packages.")
        print("The test will proceed assuming the audio files exist.")
    except Exception as e:
        print(f"Error creating dummy files: {e}. Proceeding assuming files exist.")
    
    run_tests()