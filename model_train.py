import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
import pickle

# 加载数据
data = pd.read_csv('./data/data_new.csv',index_col=0)
# 求皮尔相关系数（找重要特征）
corr = data.corr()
# 以0.08为阈值进行筛选
feature_index = corr['IS_LOST'].drop('IS_LOST').abs()>0.08
# 筛选出出特征的名字
feature_name = feature_index.loc[feature_index].index
# ['INNET_MONTH', 'CREDIT_LEVEL', 'NO_ROAM_CDR_NUM', 'CALL_DAYS',
#       'CALLING_DAYS', 'CALLED_DAYS', 'CALL_RING', 'CALLED_RING']

#自变量【影响用户流失的关键属性】
X = data.loc[:,feature_name]
y = data.loc[:,'IS_LOST'] # 预测目标

# 正样本索引（1） - 已经流失的用户
index_positive = y.index[y==1]
# 负样本索引（0） - 留下的客户 当前比例是1:30
index_negative = np.random.choice(y.index[y==0].tolist(),y.value_counts()[1])
# 取出流失用户与留下用户的特征数据
X_positive = X.loc[index_positive,:]
X_negative = X.loc[index_negative,:]
# 取出流失用户和留下用户的结果 0/1
y_positive = y.loc[index_positive]
y_negative = y.loc[index_negative]
# 数据拼接
X = pd.concat([X_positive,X_negative],axis=0)
y = pd.concat([y_positive,y_negative],axis=0)

#------------------------
# 划分数据集
X_train,X_test,y_tran,y_test=train_test_split(X,y,test_size=0.2,stratify=y)
# 构建模型1：集成学习
# 随机森林：n_estimators默认训练100次
rfc = RandomForestClassifier(n_estimators=500)
# 导入训练数据
rfc.fit(X_train,y_tran)
# 调用模型预测结果
y_pre = rfc.predict(X_test)
# 模型评估
score = rfc.score(X_test,y_test)
print(score) #79
# 混淆矩阵
confusion_matrix(y_test,y_pre)
# -----------------------------------
# 保存模型

with open('./data/yjw_Model.pkl','wb') as f:
    pickle.dump(rfc,f)
    print('模型保存成功')
    print(classification_report(y_test,y_pre))


