import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 1. 路径设置 ==========
data_path = "output/features_summary.csv"
output_dir = "output/group_feature_analysis"
os.makedirs(output_dir, exist_ok=True)

# ========== 2. 读取特征数据 ==========
df = pd.read_csv(data_path)

# ========== 3. 定义监测点分组规则 ==========
def detect_group(filename):
    name = filename.lower()
    if "top" in name:
        return "Slope top group"
    elif "platform" in name:
        return "Platform group"
    elif "bottom" in name:
        return "Slope bottom group"
    elif "surface" in name:
        return "Slope surface group"
    else:
        return "Unknown"

df["Group"] = df["File"].apply(detect_group)

# ========== 4. 统计每组均值 ==========
group_mean = df.groupby("Group").mean(numeric_only=True)
group_std = df.groupby("Group").std(numeric_only=True)

# ========== 5. 保存统计表 ==========
group_mean.to_csv(os.path.join(output_dir, "group_mean.csv"))
group_std.to_csv(os.path.join(output_dir, "group_std.csv"))

# ========== 6. 绘制各组特征柱状图（更新版） ==========
for feature in group_mean.columns:
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=group_mean.index,
        y=group_mean[feature],
        hue=group_mean.index,  # 新增
        palette="Set2",
        legend=False           # 隐藏多余图例
    )
    plt.title(f"{feature} — Group Comparison")
    plt.ylabel(feature)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{feature}_group_comparison.png"))
    plt.close()


# ========== 7. 绘制热力图（整体特征对比） ==========
plt.figure(figsize=(10, 6))
sns.heatmap(group_mean.T, annot=True, cmap="YlGnBu")
plt.title("Feature Comparison Across Groups")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "group_feature_heatmap.png"))
plt.close()

print("✅ 分组特征分析完成！结果保存在：", output_dir)
