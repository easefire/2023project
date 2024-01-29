import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel('/Users/sam/Desktop/tensorflow_bug_reports.xlsx')

# 处理数据
contributor_data = data[['author_association', 'comments']]  # 选择特征列和目标列
contributor_data.columns = ['feature', 'y']

# 将 'feature' 列进行独热编码，转换为数值特征
contributor_data = pd.get_dummies(contributor_data, columns=['feature'])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(contributor_data.drop('y', axis=1), contributor_data['y'], test_size=0.2, random_state=42)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 进行预测
predictions = model.predict(X_test)

# 计算均方根误差（RMSE）
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'RMSE: {rmse}')

# 绘制预测结果图形
plt.scatter(X_test['feature_CONTRIBUTOR'], y_test, label='Actual Data for CONTRIBUTOR')
plt.scatter(X_test['feature_MEMBER'], y_test, label='Actual Data for MEMBER')
# 绘制预测值
plt.scatter(X_test['feature_CONTRIBUTOR'], predictions, label='Predictions for CONTRIBUTOR', marker='x')
plt.scatter(X_test['feature_MEMBER'], predictions, label='Predictions for MEMBER', marker='x')

plt.xlabel('Author Association')
plt.ylabel('Comments')
plt.legend()
plt.show()
