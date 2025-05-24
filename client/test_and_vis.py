from modules.StereoImageFilter import StereoImageFilter
from CompareTable import CompareTable
left_dir = 'C:\\Users\\Dhan\\Desktop\\Project3\\snapshot\\L'
right_dir = 'C:\\Users\\Dhan\\Desktop\\Project3\\snapshot\\R'
log_path = 'C:\\Users\\Dhan\\Desktop\\Project3\\snapshot\\tank_info_log.txt'
model_path='C:\\Users\\Dhan\\Desktop\\Project3\\best.pt'
filter = StereoImageFilter(left_dir,right_dir, log_path,model_path)
log_filtered,_,_ = filter.get_result()


table_maker = CompareTable(model_path='C:\\Users\\Dhan\\Desktop\\Project3\\best.pt')
results = table_maker.compare_table(left_dir, right_dir, log_path)

real_distance = log_filtered[['Time', 'Distance']].copy()
real_distance['reg_distance'] = None
real_distance['tri_distance'] = None

for sublist in results:
    for result in sublist:
        time = float(result['Time'])
        predicted_distance_reg = float(result['reg_pred']['distance'])  # 문자열이라 변환 필요
        predicted_distance_tri = float(result['tri_pred']['distance'])
        print(predicted_distance_reg)
        real_distance.loc[real_distance['Time'] == time, 'reg_distance'] = predicted_distance_reg
        print(real_distance.loc[real_distance['Time'] == time, 'reg_distance'])
        real_distance.loc[real_distance['Time'] == time, 'tri_distance'] = predicted_distance_tri


err_reg = abs((real_distance['reg_distance']-real_distance['Distance'])/real_distance['Distance'])*100
err_tri = abs((real_distance['tri_distance']-real_distance['Distance'])/real_distance['Distance'])*100

err_df = pd.concat([err_reg,err_tri],axis = 1)
err_df.columns = ['err_reg','err_tri']
print('error from regression    : \t',round(err_df['err_reg'].mean(),2),'\nerror from triangulation : \t',round(err_df['err_tri'].mean(),2),
      '\nnumber of attempt        :\t',len(err_df['err_reg']))
df = pd.concat([real_distance, err_df], axis=1)

# err_reg가 10보다 작으면 True, 아니면 False
df['hit'] = df['err_reg'] < 10

print(df[['hit','Distance']])

df['hit'].value_counts()




#거리별 hit 시각화===================
import pandas as pd
import matplotlib.pyplot as plt

# 1. 거리 구간 정의
bins = [0, 50, 100, 150, float('inf')]
labels = ['0-50m', '50-100m', '100-150m', '150m']

# 2. 거리 구간 범주화
df['Distance_range'] = pd.cut(df['Distance'], bins=bins, labels=labels, right=False)

# 3. 각 구간별 hit=True 비율 계산
hit_ratio = df.groupby('Distance_range')['hit'].mean()

# 4. 시각화
hit_ratio.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Hit Ratio by Distance Range')
plt.xlabel('Distance Range')
plt.ylabel('Hit Ratio (True 비율)')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
