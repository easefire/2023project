import pandas as pd
import matplotlib.pyplot as plt

# 载入数据
file_path = r'D:\Users\Liky\OneDrive - mail.dlut.edu.cn\桌面\bug分析python代码\tensorflow_bug_reports.xlsx'  # 替换为您的文件路径
tensorflow_bugs = pd.read_excel(file_path)

# 检查并处理评论数字段
tensorflow_bugs['comments'] = tensorflow_bugs['comments'].fillna(0)

# 分组计算每个贡献者的bug报告的平均评论数
avg_comments_by_author = tensorflow_bugs.groupby('author_association')['comments'].mean().sort_values()

# 绘制条形图
plt.figure(figsize=(10, 6))
avg_comments_by_author.plot(kind='bar', color='skyblue')
plt.title('Average Comments per Bug Report by Author Association')
plt.xlabel('Author Association')
plt.ylabel('Average Number of Comments')
plt.xticks(rotation=45)
plt.show()
