import pandas as pd
import matplotlib.pyplot as plt

# 载入数据
file_path = r'D:\Users\Liky\OneDrive - mail.dlut.edu.cn\桌面\bug分析python代码\tensorflow_bug_reports.xlsx'
tensorflow_bugs = pd.read_excel(file_path)

# 分析不同作者关联类型的bug报告数量
author_association_counts = tensorflow_bugs['author_association'].value_counts()

# 可视化数据
plt.figure(figsize=(10, 6))
author_association_counts.plot(kind='bar', color='skyblue')
plt.title('Bug Reports by Author Association')
plt.xlabel('Author Association')
plt.ylabel('Number of Bug Reports')
plt.xticks(rotation=45)
plt.show()
