import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, hilbert
import pywt

# === 1. 目录路径 ===
data_dir = "data"
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

# === 2. 主循环 ===
for file in csv_files:
    df = pd.read_csv(file)
    # 去除列名空格
    df.columns = [col.strip() for col in df.columns]
    time = df["Time(sec)"].values
    acc = df["Acceleration(G)"].values

    # === 3. 预处理 ===
    acc = acc - np.mean(acc)  # 去直流分量
    fs = 1 / np.mean(np.diff(time))  # 采样频率

    # === 4. 多种特征分析 ===
    # 1. 原始信号
    orig = acc

    # 2. 平滑信号（移动平均）
    window_size = 50
    smooth = np.convolve(acc, np.ones(window_size)/window_size, mode='same')

    # 3. FFT 频谱
    freqs = np.fft.rfftfreq(len(acc), d=1/fs)
    fft_vals = np.abs(np.fft.rfft(acc))

    # 4. 功率谱密度 (PSD)
    f_psd, Pxx = welch(acc, fs, nperseg=1024)

    # 5. 小波变换 (CWT)
    scales = np.arange(1, 128)
    coeffs, freqs_cwt = pywt.cwt(acc, scales, 'morl', sampling_period=1/fs)

    # 6. 包络（Hilbert 变换）
    analytic_signal = hilbert(acc)
    envelope = np.abs(analytic_signal)

    # === 5. 画图 ===
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(os.path.basename(file), fontsize=14)

    axs[0, 0].plot(time, orig)
    axs[0, 0].set_title("Original Signal")

    axs[0, 1].plot(time, smooth, color='orange')
    axs[0, 1].set_title("Smoothed Signal (Moving Avg)")

    axs[0, 2].plot(freqs, fft_vals)
    axs[0, 2].set_title("FFT Spectrum")
    axs[0, 2].set_xlim(0, fs/2)

    axs[1, 0].semilogy(f_psd, Pxx)
    axs[1, 0].set_title("Power Spectral Density (PSD)")

    im = axs[1, 1].imshow(np.abs(coeffs), extent=[time[0], time[-1], freqs_cwt[-1], freqs_cwt[0]],
                          aspect='auto', cmap='jet')
    axs[1, 1].set_title("CWT Time-Frequency")
    plt.colorbar(im, ax=axs[1, 1], orientation='vertical')

    axs[1, 2].plot(time, envelope, color='green')
    axs[1, 2].set_title("Signal Envelope (Hilbert)")

    for ax in axs.flat:
        ax.set_xlabel("Time (s)")
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()
# === 6. 保存图像 ===
    base_name = os.path.splitext(os.path.basename(file))[0]
    output_path = os.path.join("output/feature", f"{base_name}_features.png")
    plt.savefig(output_path, dpi=200)
    plt.close(fig)