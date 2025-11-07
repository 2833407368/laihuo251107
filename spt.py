import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os


def plot_time_frequency_analysis(
        signal_data,
        fs,
        title="时频",
        method='spectrogram',
        window='hann',
        nperseg=256,
        noverlap=None,
        save_dir="time_frequency_plots",
        filename="time_freq_analysis",
        show_plot=True
):
    """
    绘制信号的时频图

    Parameters:
    -----------
    signal_data : array-like
        输入信号数据
    fs : float
        采样频率 (Hz)
    title : str, optional
        图表标题
    method : str, optional
        时频分析方法: 'spectrogram', 'stft', 'cwt'
    window : str or tuple, optional
        窗函数类型
    nperseg : int, optional
        每个段的长度
    noverlap : int, optional
        段之间的重叠长度，默认是nperseg的一半
    save_dir : str, optional
        保存目录
    filename : str, optional
        保存文件名
    show_plot : bool, optional
        是否显示图表

    Returns:
    --------
    fig : matplotlib.figure.Figure
        生成的图表对象
    ax : matplotlib.axes.Axes
        坐标轴对象
    """

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 默认重叠长度
    if noverlap is None:
        noverlap = nperseg // 2

    # 计算时间轴
    n_samples = len(signal_data)
    time = np.arange(n_samples) / fs

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 绘制原始信号
    ax1.plot(time, signal_data, color='blue', linewidth=1.5)
    ax1.set_title('原始信号', fontsize=14, fontweight='bold')
    ax1.set_ylabel('幅值', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, time[-1])

    # 根据选择的方法计算时频表示
    if method == 'spectrogram':
        # 功率谱密度
        f, t, Sxx = signal.spectrogram(
            signal_data,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density'
        )
        eps = np.finfo(float).eps  # 获取浮点型的最小正值
        Z = 10 * np.log10(Sxx + eps)  # 转换为dB
        cmap = 'viridis'
        zlabel = '功率谱密度 (dB/Hz)'

    elif method == 'stft':
        # 短时傅里叶变换
        f, t, Zxx = signal.stft(
            signal_data,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap
        )
        Z = np.abs(Zxx)  # 幅度谱
        cmap = 'plasma'
        zlabel = '幅度'

    elif method == 'cwt':
        # 连续小波变换
        widths = np.arange(1, 64)
        cwtmatr = signal.cwt(signal_data, signal.ricker, widths)
        t = time
        f = np.linspace(0, fs / 2, cwtmatr.shape[0])
        Z = np.abs(cwtmatr)
        cmap = 'RdBu_r'
        zlabel = '小波系数幅度'

    else:
        raise ValueError(f"不支持的方法: {method}，请选择 'spectrogram', 'stft' 或 'cwt'")

    # 绘制时频图
    if method == 'cwt':
        im = ax2.imshow(
            Z,
            extent=[0, time[-1], 0, fs / 2],
            aspect='auto',
            origin='lower',
            cmap=cmap
        )
    else:
        im = ax2.pcolormesh(
            t,
            f,
            Z,
            cmap=cmap,
            shading='gouraud'
        )

    ax2.set_title(f'{title} ({method.upper()})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('时间 (s)', fontsize=12)
    ax2.set_ylabel('频率 (Hz)', fontsize=12)
    ax2.set_ylim(0, fs / 2)  # 只显示正频率
    ax2.grid(True, alpha=0.3)

    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label(zlabel, fontsize=12)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{filename}_{method}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"时频图已保存到: {os.path.abspath(save_path)}")

    # 显示图表
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax1, ax2



# if __name__ == "__main__":
#     # 生成测试信号
#     fs = 1000  # 采样频率
#     t = np.arange(0, 10, 1 / fs)
#
#     # 生成包含多个频率成分的信号
#     signal_data = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 50 * t)
#
#     # 添加噪声
#     signal_data += 0.5 * np.random.randn(len(t))
#
#     # 测试不同的时频分析方法
#     methods = ['spectrogram', 'stft', 'cwt']
#
#     for method in methods:
#         plot_time_frequency_analysis(
#             signal_data=signal_data,
#             fs=fs,
#             title="多频率成分信号时频分析",
#             method=method,
#             nperseg=128,
#             save_dir="time_freq_demo",
#             filename="multi_freq_signal",
#             show_plot=True
#         )