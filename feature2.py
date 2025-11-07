import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, hilbert
from scipy.stats import skew, kurtosis
import pywt

# === 1. 设置路径 ===
data_dir = "data"
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
output_dir = "output/feature2"
# === 2. 读取所有 CSV 文件 ===

for file in csv_files:
    # === 读取数据 ===
    df = pd.read_csv(file)
    # 去除列名空格
    df.columns = [col.strip() for col in df.columns]
    time = df["Time(sec)"].values
    acc = df["Acceleration(G)"].values

    # === 信号预处理 ===
    acc = acc - np.mean(acc)
    fs = 1 / np.mean(np.diff(time))  # 采样频率

    # === 信号分析 ===
    smooth = np.convolve(acc, np.ones(50)/50, mode='same')
    freqs = np.fft.rfftfreq(len(acc), d=1/fs)
    fft_vals = np.abs(np.fft.rfft(acc))
    f_psd, Pxx = welch(acc, fs, nperseg=1024)
    scales = np.arange(1, 128)
    coeffs, freqs_cwt = pywt.cwt(acc, scales, 'morl', sampling_period=1/fs)
    envelope = np.abs(hilbert(acc))

    # === 统计特征计算 ===
    features = {
        "Mean": np.mean(acc),
        "Std": np.std(acc),
        "Skewness": skew(acc),
        "Kurtosis": kurtosis(acc),
        "RMS": np.sqrt(np.mean(acc**2)),
        "Max": np.max(acc),
        "Min": np.min(acc),
        "Peak-to-Peak": np.ptp(acc),
    }

    # === 绘图 ===
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(os.path.basename(file), fontsize=14)

    axs[0, 0].plot(time, acc)
    axs[0, 0].set_title("Original Signal")

    axs[0, 1].plot(time, smooth, color='orange')
    axs[0, 1].set_title("Smoothed Signal")

    axs[0, 2].plot(freqs, fft_vals)
    axs[0, 2].set_title("FFT Spectrum")
    axs[0, 2].set_xlim(0, fs/2)

    axs[1, 0].semilogy(f_psd, Pxx)
    axs[1, 0].set_title("Power Spectral Density")

    im = axs[1, 1].imshow(np.abs(coeffs),
                          extent=[time[0], time[-1], freqs_cwt[-1], freqs_cwt[0]],
                          aspect='auto', cmap='jet')
    axs[1, 1].set_title("CWT Time-Frequency")
    plt.colorbar(im, ax=axs[1, 1])

    axs[1, 2].plot(time, envelope, color='green')
    axs[1, 2].set_title("Hilbert Envelope")

    for ax in axs.flat:
        ax.set_xlabel("Time (s)")
        ax.grid(True)

    # === 在最后一个子图添加统计特征表格 ===
    table_data = [[k, f"{v:.4f}"] for k, v in features.items()]
    axs[1, 2].table(cellText=table_data, colLabels=["Feature", "Value"],
                    loc="bottom", cellLoc="center", bbox=[0, -1.15, 1, 1])
    axs[1, 2].set_ylim(np.min(envelope), np.max(envelope)*1.1)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    # === 保存结果 ===
    base_name = os.path.splitext(os.path.basename(file))[0]
    output_path = os.path.join(output_dir, f"{base_name}_features.png")
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"✅ 已保存: {output_path}")

print("🎯 所有文件处理完成，带统计特征表的图已保存在 laihuo/output/feature2/")
