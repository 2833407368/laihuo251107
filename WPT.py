import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt  # 确保已安装：pip install pywavelets

# 设置Windows系统中常见的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题



def wavelet_packet_analysis(data, time, sampling_freq, wavelet='db4', level=4):
    """
    小波包变换分析函数（修正节点访问方式）
    """
    # 创建小波包对象
    wp = pywt.WaveletPacket(data, wavelet, mode='symmetric', maxlevel=level)

    # 获取所有节点（返回的是Node对象列表）
    nodes = wp.get_level(level, 'natural')  # 注意：这里返回的是节点对象，而非名称

    # 计算每个节点的频率范围
    freq_band = sampling_freq / (2 ** level)  # 每个频段的带宽
    freq_axes = []
    energies = []

    for i, node in enumerate(nodes):
        # 直接通过节点对象获取系数（无需再用wp[node]访问）
        coeffs = node.data  # 修正：用node.data直接获取系数

        # 计算当前节点的频率范围
        freq_start = i * freq_band
        freq_end = (i + 1) * freq_band
        freq_axes.append((freq_start, freq_end))

        # 计算该频段的能量（系数平方和）
        energy = np.sum(np.square(coeffs))
        energies.append(energy)

    # 能量归一化（计算占比）
    total_energy = np.sum(energies)
    energy_ratio = [e / total_energy for e in energies] if total_energy != 0 else []

    return freq_axes, energy_ratio, wp


# 处理data文件夹中的CSV文件
data_dir = 'data'
for filename in os.listdir(data_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_dir, filename)
        print(f"正在处理文件：{filename}")

        # 读取数据（处理带空格的列名）
        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            print(f"  读取文件失败：{e}")
            continue

        # 查找时间和加速度列（兼容带空格的列名）
        try:
            time_col = [col for col in data.columns if 'Time(sec)' in col][0]
            acc_col = [col for col in data.columns if 'Acceleration(G)' in col][0]
        except IndexError:
            print(f"  未找到时间或加速度列，跳过该文件")
            continue

        time = data[time_col].values
        acceleration = data[acc_col].values

        # 计算采样频率
        try:
            sampling_freq = 1 / np.mean(np.diff(time))
            print(f"  采样频率：{sampling_freq:.2f} Hz")
        except Exception as e:
            print(f"  计算采样频率失败：{e}")
            continue

        # 去除直流分量
        acceleration = acceleration - np.mean(acceleration)

        # 小波包分析（4层分解，使用db4小波基）
        try:
            freq_bands, energy_ratios, wp = wavelet_packet_analysis(
                acceleration, time, sampling_freq,
                wavelet='db4', level=4
            )
        except Exception as e:
            print(f"  小波包分析失败：{e}")
            continue

        # 提取能量占比最高的频段（主频段）
        if energy_ratios:
            max_energy_idx = np.argmax(energy_ratios)
            main_band = freq_bands[max_energy_idx]
            print(f"  主频段：{main_band[0]:.2f} - {main_band[1]:.2f} Hz "
                  f"(能量占比：{energy_ratios[max_energy_idx]:.2%})")
        else:
            print("  无法计算能量占比（总能量为0）")

        # 可视化结果
        try:
            plt.figure(figsize=(12, 6))

            # 1. 原始信号图
            plt.subplot(2, 1, 1)
            plt.plot(time, acceleration)
            plt.xlabel('时间 (秒)')
            plt.ylabel('加速度 (G)')
            plt.title(f'{filename} 原始信号')
            plt.grid(True)

            # 2. 小波包能量谱
            plt.subplot(2, 1, 2)
            band_centers = [(b[0] + b[1]) / 2 for b in freq_bands]
            plt.bar(band_centers, energy_ratios, width=freq_bands[0][1] - freq_bands[0][0] * 0.8)
            plt.xlabel('频率 (Hz)')
            plt.ylabel('能量占比')
            plt.title(f'小波包能量谱（主频段：{main_band[0]:.2f}-{main_band[1]:.2f} Hz）')
            plt.grid(True, axis='y')

            plt.tight_layout()
            plt.savefig(f"output/WPT/{filename}_小波包分析.png")
            plt.close()

            # 【调用时频分析函数】
            import spt
            spt.plot_time_frequency_analysis(
                signal_data=acceleration,  # 加速度
                fs=sampling_freq,  # 使用计算出的采样频率
                title=f"{filename} 时频分析",
                method="stft",  # 可选择'spectrogram'/'stft'/'cwt'
                nperseg=128,
                save_dir="output/时频图/WPT",
                filename=f"{filename}",
                show_plot=False
            )

        except Exception as e:
            print(f"  绘图失败：{e}")
            continue

print("ok")