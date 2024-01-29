import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel('/Users/sam/Desktop/tensorflow_bug_reports.xlsx')

# 处理数据
bug_resolve_data = data[['created_at', 'updated_at', 'number']]  # 选择特征列和目标列
bug_resolve_data.columns = ['feature1', 'feature2', 'y']

# 使用 .loc 避免 SettingWithCopyWarning
bug_resolve_data.loc[:, 'feature1'] = pd.to_datetime(bug_resolve_data['feature1']).astype(int) // 10**9  # 转换为秒
bug_resolve_data.loc[:, 'feature2'] = pd.to_datetime(bug_resolve_data['feature2']).astype(int) // 10**9  # 转换为秒

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(bug_resolve_data[['feature1', 'feature2']], bug_resolve_data['y'], test_size=0.2, random_state=42)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 进行预测
predictions = model.predict(X_test)

# 计算均方根误差（RMSE）
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'RMSE: {rmse}')

# 绘制预测结果图形
plt.scatter(X_test['feature1'], y_test, label='Actual Data')
plt.scatter(X_test['feature1'], predictions, label='Predictions')
plt.xlabel('Feature 1')
plt.ylabel('Bug Resolve Time')
plt.legend()
plt.show()
