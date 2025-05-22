import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

result_path = './hit_rate_log.csv' # /hit_rate_angle_vs_coord.csv
output_dir = 'report'

# 출력 디렉토리 없으면 생성
os.makedirs(output_dir, exist_ok=True)

# 데이터 로드
df = pd.read_csv(result_path)

# algorithm 별 평균 및 표준 편차 계산
summary= df.groupby('algorithm')['hit_rate'].agg(['mean', 'std']).reset_index()
summary.columns = ['algorithm', 'mean_hit_rate', 'std_hit_rate']

print("명중률 통계 - 평균, 표준편차")
print("summary", summary)

# 그래프
sns.set(style="whitegrid")
palette = sns.color_palette("Set2")

plt.figure(figsize=(12, 6))
sns.barplot(x='algorithm', y='mean_hit_rate', data=summary, palette=palette, edgecolor ='black', capsize=.2)
plt.errorbar(x=summary['algorithm'], y=summary['mean_hit_rate'], yerr=summary['std_hit_rate'], fmt='none', capsize=5, color='black')

plt.title('Hit Rate by Matching Method', fontsize=15, weight='bold')
plt.xlabel('Maching Method', fontsize=12)
plt.ylabel('Mean Hit Rate', fontsize=12)
plt.ylim(0, 1.0)
plt.tight_layout()

# 그래프 저장
plot_path = os.path.join(output_dir, 'hit_rate_by_algorithm.png')
plt.savefig(plot_path, dpi = 300)
print(f"Hit rate graph saved to {plot_path}")

# 요약 텍스트 저장
summary_text = os.path.join(output_dir, 'summary.txt')
with open(summary_text, 'w') as f:
     for _, row in summary.iterrows():
        f.write(f"Algorithm: {row['algorithm']}, Mean Hit Rate: {row['mean_hit_rate']:.2f}, Standard Deviation: {row['std_hit_rate']:.2f}\n")

print(f"Summary saved to {summary_text}")