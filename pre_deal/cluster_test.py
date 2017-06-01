# coding=utf-8
import pandas as pd
from  pandas import Series, DataFrame
import matplotlib.pyplot as  plt
import time
import  sklearn
from sklearn.cluster import KMeans
import  numpy as np

# start =time.clock()
# end = time.clock()
# print('Running time: %s Seconds'%(end-start))

def pre_deal():
    try:
        ad = pd.read_csv(r'E:\dataset\ad.csv', sep=',')  # 只能读取英文路径,返回的是一个dataframe类型数据
        app_cate = pd.read_csv(r'E:\dataset\app_categories.csv', sep=',')
        # tr = pd.read_csv(r'E:\dataset\train.csv', sep = ',')
        position = pd.read_csv(r'E:\dataset\position.csv', sep=',')
        user_info = pd.read_csv(r'E:\dataset\user.csv', sep=',')
        #
        # user_installedapps = pd.read_csv(r'E:\dataset\user_installedapps.csv', sep = ',')#读取的最大的文件，尽可能避免处理此文件
        # user_installedapps = pd.read_csv(r'E:\dataset\position.csv', sep=',')#假的
        # user_app_actions = pd.read_csv(r'E:\dataset\user_app_actions.csv', sep=',')
        # train3 = pd.read_csv(r'E:\dataset\train3.csv', sep = ',')
        # train4 = pd.read_csv(r'E:\dataset\train4.csv', sep=',')

        test = pd.read_csv(r'E:\dataset\test.csv', sep=',')


    except:
        print "打开文件的时候出错!"

def cluster():
    try:
        test_2 = pd.read_csv(r'E:\dataset\test2.csv', sep=',')  # 只能读取英文路径,返回的是一个dataframe类型数据
    except:
        print "打开文件的时候出错!"


    x = test_2.drop(['instanceID','label'],1)
    y_pred = KMeans(n_clusters = 3, random_state = 9).fit_predict(x)
    np.savetxt('test2_y.csv', y_pred,fmt='%d', delimiter=',',header='#')
    #需要输出聚类的结果
def cal():
    try:
        test_y = pd.read_csv(r'test2_y.csv', sep=',')  # 只能读取英文路径,返回的是一个dataframe类型数据
    except:
        print "打开文件的时候出错!"
    test_y.columns = ['label']
    print test_y.info()
    print test_y.label.value_counts()

if __name__ == '__main__':
    try:
        # train4 = pd.read_csv(r'E:\dataset\train4.csv', sep=',')
        train5 = pd.read_csv(r'train5.csv', sep=',')

    except:
        print "打开文件的时候出错!"
    t =  train5.adID.value_counts()
    t.to_csv('adID.csv', index=False, sep=',')
    # cluster()
    # cal()


