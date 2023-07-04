import pandas as pd
import pickle

# 加载数据
data = pd.read_csv('./data/data_p.csv',index_col=0)

# 加载模型
with open('./model/YYS_Model.pkl','rb') as f:
    model = pickle.load(f)

# 预测数据
pred_y = model.predict(data[['INNET_MONTH', 'CREDIT_LEVEL', 'NO_ROAM_CDR_NUM', 'CALL_DAYS',
                            'CALLING_DAYS', 'CALLED_DAYS', 'CALL_RING', 'CALLED_RING']])
print(pred_y)
#-----------------------------------------
'''
拓展可迭代：
    替换其他模型训练：决策时、支持向量机、K近邻等
    QT:制作界面
同类型项目：
    体检报告：
    银行用户风险：
    分类：
'''