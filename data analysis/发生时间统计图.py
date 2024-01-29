
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 载入数据
file_path = r'D:\Users\Liky\OneDrive - mail.dlut.edu.cn\桌面\bug分析python代码\tensorflow_bug_reports.xlsx'
tensorflow_bugs = pd.read_excel(file_path)

# 确保 'created_at' 字段是日期时间格式
tensorflow_bugs['created_at'] = pd.to_datetime(tensorflow_bugs['created_at'], errors='coerce')

# 提取日期时间中的年份和月份
tensorflow_bugs['year_month'] = tensorflow_bugs['created_at'].dt.to_period('M')

# 计算每个年月的bug报告数量
bug_counts_by_month = tensorflow_bugs['year_month'].value_counts().sort_index()

# 绘制图表
plt.figure(figsize=(15, 6))
bug_counts_by_month.plot(kind='bar', color='skyblue', alpha=0.6)

# 计算趋势线
x = np.arange(len(bug_counts_by_month))
y = bug_counts_by_month.values
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

# 绘制趋势线
plt.plot(x, p(x), "r--")

# 设置图表标题和轴标签
plt.title('Bug Reports by Month with Trend Line')
plt.xlabel('Year-Month')
plt.ylabel('Number of Bug Reports')
plt.xticks(rotation=45, ticks=np.arange(len(bug_counts_by_month)), labels=bug_counts_by_month.index)
plt.show()

