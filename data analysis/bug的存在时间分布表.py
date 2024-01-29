import pandas as pd
import matplotlib.pyplot as plt

# 载入数据
file_path = r'D:\Users\Liky\OneDrive - mail.dlut.edu.cn\桌面\bug分析python代码\tensorflow_bug_reports.xlsx' # 替换为您的文件路径
tensorflow_bugs = pd.read_excel(file_path)

# 确保 'created_at' 和 'updated_at' 字段是日期时间格式
tensorflow_bugs['created_at'] = pd.to_datetime(tensorflow_bugs['created_at'], errors='coerce')
tensorflow_bugs['updated_at'] = pd.to_datetime(tensorflow_bugs['updated_at'], errors='coerce')

# 使用最后一次更新时间作为估计的关闭时间
tensorflow_bugs['estimated_resolution_time'] = tensorflow_bugs['updated_at'] - tensorflow_bugs['created_at']

# 将时间差转换为天数
tensorflow_bugs['estimated_resolution_days'] = tensorflow_bugs['estimated_resolution_time'].dt.days

# 过滤掉负值和异常大的值
tensorflow_bugs = tensorflow_bugs[tensorflow_bugs['estimated_resolution_days'] >= 0]
tensorflow_bugs = tensorflow_bugs[tensorflow_bugs['estimated_resolution_days'] < tensorflow_bugs['estimated_resolution_days'].quantile(0.95)]

# 绘制条形图
plt.figure(figsize=(12, 6))
tensorflow_bugs['estimated_resolution_days'].plot(kind='hist', bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Estimated Bug Resolution Time (Days)')
plt.xlabel('Estimated Resolution Time (Days)')
plt.ylabel('Number of Bugs')
plt.show()
