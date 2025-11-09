import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ======== 参数设置 ========
input_file = "af/all_features.csv"
output_dir = "af/feature_plots1"
os.makedirs(output_dir, exist_ok=True)

# ======== 读取特征表 ========
df = pd.read_csv(input_file)

# 去掉 File 列，只保留数值型特征
features = [col for col in df.columns if col != "File"]

# ======== 1. 各特征柱状图 ========
for feature in features:
    plt.figure(figsize=(10, 5))
    sns.barplot(x="File", y=feature, data=df, palette="Set2", hue="File", legend=False)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{feature} Across Files")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{feature}_bar.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 已保存柱状图: {save_path}")

# ======== 2. 箱线图（整体特征分布） ========
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[features], palette="Set3")
plt.title("Feature Value Distributions")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "boxplot_all_features.png"), dpi=300)
plt.close()
print("✅ 已保存箱线图")

# ======== 3. 热力图（特征相关性） ========
corr = df[features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300)
plt.close()
print("✅ 已保存特征相关性热力图")

print(f"\n🎨 所有图已保存到: {output_dir}")
