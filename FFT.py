import os
import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib
import spt

import matplotlib.pyplot as plt

# 设置Windows系统中常见的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据文件夹路径
data_dir = 'data'

# 遍历 data 文件夹中的所有 CSV 文件
for filename in os.listdir(data_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_dir, filename)
        print(f"正在处理文件：{filename}")

        # 读取数据
        data = pd.read_csv(file_path)

        # 提取时间和加速度数据（注意列名的空格）
        time_col = [col for col in data.columns if 'Time(sec)' in col][0]
        acc_col = [col for col in data.columns if 'Acceleration(G)' in col][0]
        time = data[time_col].values
        acceleration = data[acc_col].values

        # 计算采样频率
        sampling_frequency = 1 / np.mean(np.diff(time))

        # 去除直流分量
        acceleration = acceleration - np.mean(acceleration)

        # 计算 FFT
        n = len(time)
        fft_result = np.fft.fft(acceleration)
        amplitude = np.abs(fft_result)[:n // 2] * 2 / n  # 幅值归一化

        # 生成频率轴
        frequencies = np.fft.fftfreq(n, 1 / sampling_frequency)[:n // 2]

        # 提取主频
        main_frequency = frequencies[np.argmax(amplitude)]
        print(f"  信号主频为：{main_frequency:.2f} Hz")

        # 可视化频谱
        plt.figure(figsize=(10, 4))
        plt.plot(frequencies, amplitude)
        plt.xlabel('频率 (Hz)')
        plt.ylabel('幅值')
        plt.title(f'{filename} 信号频谱（主频：{main_frequency:.2f} Hz）')
        plt.grid(True)
        plt.xlim(0, 100)  # 只显示 0-100Hz 范围
        plt.tight_layout()
        plt.savefig(f"output/FTP/{filename}_频谱.png")
        plt.close()  # 关闭图形以释放内存

        # 【调用时频分析函数】
        # 注意：传递加速度数据（一维数组）而非原始DataFrame
        spt.plot_time_frequency_analysis(
            signal_data=acceleration,  #加速度信号
            fs=sampling_frequency,  #采样频率
            title=f"{filename} 时频图",
            method="spectrogram",
            nperseg=128,
            save_dir="output/时频图/FFT",
            filename=f"{filename}",
            show_plot=False
        )

print("所有文件处理完成！")