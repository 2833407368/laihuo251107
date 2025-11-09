import os
import numpy as np
import pandas as pd
from scipy.signal import welch, hilbert
from scipy.stats import kurtosis, skew
from scipy.fft import rfft, rfftfreq
from scipy.integrate import trapezoid
import antropy as ant  # pip install antropy
from hurst import compute_Hc  # pip install hurst

# ========== 特征计算函数 ==========
def compute_features(time, acc):
    fs = 1 / np.mean(np.diff(time))  # 采样频率
    features = {}

    # --- 基础统计 ---
    features["Mean"] = np.mean(acc)
    features["Std"] = np.std(acc)
    features["RMS"] = np.sqrt(np.mean(acc ** 2))
    features["Peak_to_Peak"] = np.ptp(acc)
    features["Skewness"] = skew(acc)
    features["Kurtosis"] = kurtosis(acc)
    features["Energy_Intensity"] = np.sum(acc ** 2)

    # --- 频谱特征 ---
    f, Pxx = welch(acc, fs=fs, nperseg=min(256, len(acc)))
    features["Spectral_Centroid"] = np.sum(f * Pxx) / np.sum(Pxx)
    low_energy = trapezoid(Pxx[f < fs / 4])
    high_energy = trapezoid(Pxx[f >= fs / 4])
    features["PSD_LowHigh_Ratio"] = low_energy / (high_energy + 1e-8)

    # --- Hilbert 包络 ---
    envelope = np.abs(hilbert(acc))
    features["Hilbert_Max"] = np.max(envelope)

    # --- FFT 主频 ---
    yf = np.abs(rfft(acc))
    xf = rfftfreq(len(acc), 1 / fs)
    features["FFT_Main_Freq"] = xf[np.argmax(yf)]

    # --- 复杂特征 ---
    try:
        features["Sample_Entropy"] = ant.sample_entropy(acc)
    except Exception:
        features["Sample_Entropy"] = np.nan

    try:
        features["Hurst_Exponent"], _, _ = compute_Hc(acc, kind='price', simplified=True)
    except Exception:
        features["Hurst_Exponent"] = np.nan

    try:
        features["Fractal_Dimension"] = ant.higuchi_fd(acc)
    except Exception:
        features["Fractal_Dimension"] = np.nan

    try:
        features["Lyapunov_Exponent"] = ant.lyap_r(acc)
    except Exception:
        features["Lyapunov_Exponent"] = np.nan

    # --- Teager 能量算子 ---
    teo = acc[1:-1] ** 2 - acc[:-2] * acc[2:]
    features["Teager_Energy_Mean"] = np.mean(teo)
    features["Teager_Energy_Max"] = np.max(teo)

    return features


# ========== 主程序 ==========
data_dir = "data"
output_file = "af/all_features.csv"

all_results = []

for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)

        # 自动匹配列名
        time_col = [c for c in df.columns if "time" in c.lower()][0]
        acc_col = [c for c in df.columns if "acc" in c.lower()][0]

        time = df[time_col].values
        acc = df[acc_col].values

        feats = compute_features(time, acc)
        feats["File"] = file
        all_results.append(feats)
        print(f"✅ 已处理: {file}")

# 保存结果
df_all = pd.DataFrame(all_results)
df_all.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n✅ 所有文件特征已保存至: {output_file}")
