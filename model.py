# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split

start_time = time.time()

#读入数据
train = pd.read_csv('/Users/xuxian/doing/Tpai/mytrain1.csv')
tests = pd.read_csv('/Users/xuxian/doing/Tpai/test2.csv')

train_xy,val = train_test_split(train, test_size=0.2, random_state=1)

y = train_xy.label
X = train_xy.drop(['label'],axis=1)
val_y = val.label
val_X = val.drop(['label'],axis=1)

#xgb矩阵赋值
xgb_val = xgb.DMatrix(val_X,label=val_y)
xgb_train = xgb.DMatrix(X, label=y)
test = tests.drop(['instanceID','label'],axis=1)
xgb_test = xgb.DMatrix(test)


#
params={
'booster':'gbtree',
'objective': 'binary:logistic', #二分类的问题
'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':12, # 构建树的深度，越大越容易过拟合
'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样
'min_child_weight':2,
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.05, # 如同学习率
'seed':1000,
# 'nthread':4,# cpu 线程数
'eval_metric': 'logloss'
}
plst = list(params.items())
num_rounds = 500 # 迭代次数
watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]

#训练模型并保存
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练

model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=50)
model.save_model('./xgb2.model') # 用于存储训练出的模型
print "best best_ntree_limit",model.best_ntree_limit
#
# bst = xgb.Booster()
# bst.load_model('./xgb.model')

preds = model.predict(xgb_test)
out_df = pd.DataFrame(preds)
out_df.columns = ["prob"]
out_df["instanceID"] = tests.instanceID.values
out_df[["instanceID","prob"]].to_csv("submission.csv", index=False)
#
# preds = model.predict(xgb_test,ntree_limit=model.best_ntree_limit)
# np.savetxt('xgb_submission.csv',np.c_[range(1,len(tests)+1),preds],delimiter=',',header='ImageId,Label',comments='',fmt='%d')

#输出运行时长
cost_time = time.time()-start_time
print "xgboost success!",'\n',"cost time:",cost_time,"(s)......"