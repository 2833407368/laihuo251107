import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.signal import welch, hilbert

# ----------------------
# 1. 路径设置
# ----------------------
data_dir = "data"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# ----------------------
# 2. 特征计算函数
# ----------------------
def extract_features(time, acc):
    # 基础指标
    mean_val = np.mean(acc)
    std_val = np.std(acc)
    rms = np.sqrt(np.mean(acc**2))
    peak_to_peak = np.max(acc) - np.min(acc)
    kurt = kurtosis(acc)
    skewness = skew(acc)

    # PSD 能量比（低频/高频能量）
    f, Pxx = welch(acc, fs=1/np.mean(np.diff(time)), nperseg=1024)
    low_freq_energy = np.sum(Pxx[f < np.median(f)])
    high_freq_energy = np.sum(Pxx[f >= np.median(f)])
    psd_ratio = low_freq_energy / (high_freq_energy + 1e-9)

    # Spectral Centroid（频谱质心）
    spectral_centroid = np.sum(f * Pxx) / np.sum(Pxx)

    # Hilbert 包络最大值
    analytic_signal = hilbert(acc)
    envelope = np.abs(analytic_signal)
    hilbert_max = np.max(envelope)

    # RMS 能量强度（与振动强度直接相关）
    vibration_energy = np.mean(acc**2)

    return {
        "Mean": mean_val,
        "Std": std_val,
        "RMS": rms,
        "VibrationEnergy": vibration_energy,
        "Kurtosis": kurt,
        "Skewness": skewness,
        "SpectralCentroid": spectral_centroid,
        "PeakToPeak": peak_to_peak,
        "PSD_LowHigh_Ratio": psd_ratio,
        "HilbertEnvelopeMax": hilbert_max
    }

# ----------------------
# 3. 遍历所有 CSV 文件
# ----------------------
summary = []
for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"无法读取 {filename}: {e}")
            continue

        # 列名修正（去空格）
        df.columns = [c.strip() for c in df.columns]
        time = df["Time(sec)"].values
        acc = df["Acceleration(G)"].values

        features = extract_features(time, acc)
        features["File"] = filename
        summary.append(features)

        # 绘图
        plt.figure(figsize=(10, 5))
        plt.plot(time, acc, label="Acceleration Signal", color='steelblue')
        plt.title(f"Signal and Hilbert Envelope — {filename}")
        plt.xlabel("Time (sec)")
        plt.ylabel("Acceleration (G)")
        plt.grid(True, alpha=0.3)

        # Hilbert 包络叠加
        analytic_signal = hilbert(acc)
        envelope = np.abs(analytic_signal)
        plt.plot(time, envelope, color='orange', label="Hilbert Envelope", alpha=0.8)

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_features.png"))
        plt.close()

# ----------------------
# 4. 保存汇总结果
# ----------------------
if summary:
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(os.path.join(output_dir, "features_summary.csv"), index=False)
    print(f"✅ 特征已保存至 {output_dir}/features_summary.csv")
else:
    print("⚠️ 未找到任何 CSV 文件或计算结果为空。")
