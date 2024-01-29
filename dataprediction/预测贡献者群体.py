import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 选择需要的列
selected_columns = ['number', 'title', 'user', 'labels', 'state', 'assignee', 'milestone', 'comments', 'created_at', 'updated_at', 'closed_at', 'author_association', 'body', 'reactions', 'draft', 'pull_request']

# 读取数据
file_path = "/Users/sam/Desktop/tensorflow_bug_reports.xlsx"
df = pd.read_excel(file_path)

# 划分训练集和测试集
X = df[selected_columns]
y = df['author_association']  # 使用 'author_association' 列作为目标列

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 获取数值特征和分类特征的列索引
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# 数值特征处理
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# 分类特征处理
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 组合处理结果
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough')

# 定义模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 构建Pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', model)])

# 训练模型
pipeline.fit(X_train, y_train)

# 预测并评估模型
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 打印分类报告
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)
