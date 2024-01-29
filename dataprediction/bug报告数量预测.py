import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 读取数据
data = pd.read_excel('/Users/sam/Downloads/bug分析python代码/tensorflow_bug_reports.xlsx')
bug_count_data = data[['created_at', 'id']]
bug_count_data.columns = ['ds', 'y']

# 将日期字段转换为datetime类型
bug_count_data['ds'] = pd.to_datetime(bug_count_data['ds'])

# 划分训练集和测试集
train_size = int(len(bug_count_data) * 0.8)
train, test = bug_count_data[:train_size], bug_count_data[train_size:]

# 使用指数平滑模型
model = ExponentialSmoothing(train['y'], seasonal='add', seasonal_periods=12)
result = model.fit()

# 进行预测
predictions = result.predict(start=test.index[0], end=test.index[-1])

# 绘制预测结果
plt.plot(train['ds'], train['y'], label='Training Data')
plt.plot(test['ds'], test['y'], label='Test Data')
plt.plot(test['ds'], predictions, label='Predictions')
plt.legend()
plt.show()

