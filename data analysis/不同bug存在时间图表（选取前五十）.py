import pandas as pd
import matplotlib.pyplot as plt

# 载入数据
file_path = r'D:\Users\Liky\OneDrive - mail.dlut.edu.cn\桌面\bug分析python代码\tensorflow_bug_reports.xlsx' # 替换为您的文件路径
tensorflow_bugs = pd.read_excel(file_path)

# 处理数据，计算bug的存在时间
tensorflow_bugs['created_at'] = pd.to_datetime(tensorflow_bugs['created_at'], errors='coerce')
tensorflow_bugs['updated_at'] = pd.to_datetime(tensorflow_bugs['updated_at'], errors='coerce')
tensorflow_bugs['estimated_resolution_time'] = tensorflow_bugs['updated_at'] - tensorflow_bugs['created_at']
tensorflow_bugs['estimated_resolution_days'] = tensorflow_bugs['estimated_resolution_time'].dt.days

# 选取前50个存在时间最长的bug
top_50_bugs = tensorflow_bugs.nlargest(50, 'estimated_resolution_days')

# 绘制条形图
plt.figure(figsize=(10, 12))
for i, (index, row) in enumerate(top_50_bugs.iterrows()):
    plt.barh(i, row['estimated_resolution_days'], color='skyblue')
    plt.text(row['estimated_resolution_days'], i, f'ID: {index}', va='center')

plt.xlabel('Estimated Resolution Time (Days)')
plt.ylabel('Bugs')
plt.title('Top 50 Bugs with Longest Estimated Resolution Time')
plt.yticks(range(50), top_50_bugs['title'].apply(lambda x: x[:15] + '...'))
plt.gca().invert_yaxis()
plt.show()
