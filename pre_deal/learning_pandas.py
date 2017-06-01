#coding=utf-8
import pandas as pd
from  pandas import  Series,DataFrame
import matplotlib.pyplot as  plt
import time
from sklearn.preprocessing import LabelEncoder
# start =time.clock()
# end = time.clock()
# print('Running time: %s Seconds'%(end-start))
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np



def pre_deal():
    if(0):
        tr_te = pd.read_csv(r'E:\dataset\tr_te_25_1.csv', sep=',')
        tr_te = tr_te.drop('0',1)
    else:
        try:
            ad = pd.read_csv(r'E:\dataset\ad.csv', sep=',')  # 只能读取英文路径,返回的是一个dataframe类型数据
            # app_cate = pd.read_csv(r'E:\dataset\app_cate_onehot.csv',sep = ',')
            app_cate = pd.read_csv(r'E:\dataset\app_categories.csv', sep=',')
            tr = pd.read_csv(r'E:\dataset\train.csv', sep=',')
            position = pd.read_csv(r'E:\dataset\position.csv', sep=',')
            user_info = pd.read_csv(r'E:\dataset\user.csv', sep=',')
            # ui = pd.read_csv(r'user_install3.csv', sep = ',')
            #
            # user_installedapps = pd.read_csv(r'E:\dataset\user_installedapps.csv', sep = ',')#读取的最大的文件，尽可能避免处理此文件
            # user_installedapps = pd.read_csv(r'E:\dataset\position.csv', sep=',')#假的
            # user_app_actions = pd.read_csv(r'E:\dataset\user_app_actions.csv', sep=',')
            # train3 = pd.read_csv(r'E:\dataset\train3.csv', sep = ',')
            # train4 = pd.read_csv(r'E:\dataset\train4.csv', sep=',')

            test = pd.read_csv(r'E:\dataset\test.csv', sep=',')
            # train_time_cluster = pd.read_csv(r'E:\dataset\train_time_cluster.csv', sep=',')
            # test_time_cluster = pd.read_csv(r'E:\dataset\test_time_cluster.csv', sep=',')
            # train_app_cluster = pd.read_csv(r'E:\dataset\train_app_cluster.csv', sep=',')
            # test_app_cluster = pd.read_csv(r'E:\dataset\test_app_cluster.csv', sep=',')
            # y  = pd.read_csv(r'y.csv', sep=',')
            # ytr = pd.read_csv(r'ytr.csv', sep=',')
            # train3 = pd.read_csv(r'E:\dataset\train_23_1.csv', sep=',')
            # test3 = pd.read_csv(r'E:\dataset\test_23_1.csv', sep=',')
            # ui_sum = pd.read_csv(r'E:\dataset\ui_sum.csv', sep=',')
        except:
            print "--" * 20 + "打开文件的时候出错!" * 10 + "--" * 20
    ###############################特征最多的文件属性
    start = time.clock()
    start_from_cluster = 2
    #在已经做过的几个聚类的基础上开始
    if(start_from_cluster - 1):
        if (0):
            # train3 = pd.merge(train3, ui_sum, on='userID', how="left")
            # test3 = pd.merge(test3, ui_sum, on='userID', how="left")

            return 1
            # train_id = train3.loc[:, ['gender', 'appID_rate']]
            # test_id = test3.loc[:, ['appID_rate', 'clickTime']]
            # train3['train_cluster'] = test_kmeans_cluster(train_id)
            # test3['train_cluster'] = test_kmeans_cluster(test_id)
            # 测试新特征效果


            se = 'sum'

            tr = train3[['label', se]]
            tr1 = DataFrame(tr)
            # tr1['ft'] = train3['train_cluster']
            # tr1 = tr1.sort(columns='train_cluster')
            # tr2 = tr1.groupby(['train_cluster']).sum()
            # tr3 = tr1.train_cluster.value_counts()
            ######################
            tr1 = tr1.sort(columns=se)
            tr2 = tr1.groupby([se]).sum()
            tr3 = tr1[se].value_counts()  # 统计每个值出现的次数

            tr4 = pd.concat([tr2, tr3], axis=1)

            tr4.to_csv('feature_test.csv', index=False, sep=',')
            return 1
            # 将test和train合并，test添加conversionTime列，train添加instanceID，共9列
        print  "--" * 20 + "正在合并test和train文件" + "--" * 20
        if (1):
            tr_te = get_concat(tr, test)

        ad1 = deal_ad(ad, app_cate)
        tr_te_first_deal = deal_ad1(tr_te, ad1)
        # train_first_deal = deal_ad1(tr, ad1)

        print "--" * 20 + "正在处理用户信息！" + "--" * 20
        tr_te_second_deal = deal_user_info(user_info, tr_te_first_deal)
        # train_second_deal = deal_user_info(user_info, train_first_deal)

        print "--" * 20 + "正在处理广告位置信息！" + "--" * 20
        tr_te_3th_deal = deal_position(position, tr_te_second_deal)
        # train_3th_deal = deal_position(position, train_second_deal)
        print "--" * 20 + "正在处理用户安装信息！" + "--" * 20

        ##改变前两列的顺序
        # test_3th_deal = change_seq(test_3th_deal)
        # train_3th_deal = change_seq(train_3th_deal)
        # print test_3th_deal.info()
        # print train_3th_deal.info()
        # 有 一些ID的信息无法统计
        # test_3th_deal = deal_ui(ui, test_3th_deal)
        # train_3th_deal = deal_ui(ui, train_3th_deal)

        tr_te = tr_te_3th_deal
        # train3 = train_3th_deal


        #############################取出激活用户##################################
        # df = train3[train3['label'] == 1]
        # df.to_csv('train_17_1_label.csv', index=False, sep=',')
        # return 1

        # 加入时间
        if (True):
            # train3, test3 = ensure_sequence(train3, test3)
            lables = [1.1, 2, 3, 4, 5, 6, 7, 1.2]
            tr_te_ = tr_te.clickTime / 10000
            # test3_ = pd.cut(test3_, [0,1, 7, 12, 14, 18, 22,23,24], right=False, labels=lables)
            tr_te_ = tr_te_.astype('int')
            # test3 = test3.sort(columns = 'instanceID')
            ##########对train的时间处理
            # train3_ = train3.clickTime / 10000
            # train3_ = pd.cut(train3_, [0,1, 7, 12, 14, 18, 22, 23,24], right=False, labels=lables)
            # train3_ = train3_.astype('int')

            # train3, test3 = ensure_sequence(train3,test3)

            tr_te['day'] = tr_te_
            tr_te['day'] = tr_te_
            # 给时间区间重新分段
            # train3=train3.fillna(0)
            # test3=test3.fillna(0)
            # return 1
        # 将几个id类特征利用统计的次数表示,appID,camgaignID,advertiserID，    creativeID,adID，

        # t2 = pd.concat([train3,test3])

        if (0):
            # train3, test3 = ensure_sequence(train3, test3)
            list1 = ['appID', 'camgaignID', 'advertiserID', 'creativeID', 'adID']

            print '--' * 20, "正在处理历史点击率", '--' * 20
            for li in list1:
                te = get_count(tr_te, li)
                tr_te = deal_rate(tr_te, te, li)
                # test3 = deal_rate(test3, te, li)

                # test3['advertiser_id_cnt2'] = np.minimum(test3.cnt_advertiser_id.astype('int32').values, 3000)  # 出现次数大于三百的用300表示
                # test3['cnt_ad_id'] = get_agg(test3.adID.values, test3.instanceID, np.size)
                # train3, test3 = ensure_sequence(train3, test3)

                # test3 = test3.drop(['cnt_app_id', 'cnt_camgaign_id', 'cnt_advertiser_id'], 1)  # 删除train3.csv中的一些列
                # train3 = train3.drop(['cnt_app_id', 'cnt_camgaign_id', 'cnt_advertiser_id'], 1)  # 删除train3.csv中的一些列\
                # 聚类增加特征


    print '--' * 20, "正在进行聚类增加特征！", '--' * 20
    if(0):
        #adID,camgaignID,advertiserID,appPlatform,appCategory,age,gender,education,clickTime
        #connectionType,telecomsOperator,adID,camgaignID,advertiserID,appID,appPlatform,appCategory,age,gender,education,marriageStatus,haveBaby,hometown,residence,sitesetID,positionType
        list = [['positionType', 'connectionType'], ['advertiserID','positionType'], ['gender','positionType'],
                 ['hometown','residence'],['age','marriageStatus'],['age','positionType'],
                 ['appCategory','positionType'],['hometown','positionType'],['positionType','telecomsOperator'],
                 ['creativeID','positionType'],['education','positionType'],['camgaignID','positionType'],
                 ['age','creativeID']]
        list_cluster = [['adID','gender'],['creativeID','gender'],['camgaignID','gender'],['appID','gender'],
                        ['marriageStatus','positionType'],['residence','positionType'],['age','telecomsOperator'],
                       ]

        for clu in list_cluster:
            tr_te_id = tr_te.loc[:, clu]
            # test_id = tr_te.loc[:, clu]
            tr_te['cluster_'+clu[0]+'_'+clu[1]] = test_kmeans_cluster(tr_te_id)
            # tr_te['cluster_'+clu[0]+'_'+clu[1]] = test_kmeans_cluster(tr_te_id)

    tr_te.to_csv(r'E:\dataset\tr_te_cluster_25_1.csv',index=False,sep = ',')
    start_from_cluster = 0
    if(start_from_cluster):
        filter_test = np.logical_and(tr_te.day.values > 30, 1)
        filter_train = np.logical_and(tr_te.day.values < 30, tr_te.day.values > 16)
        filter_va = np.logical_and(tr_te.day.values == 30, 1)
        # 训练集,测试集

        train = tr_te.ix[filter_train, :].copy()
        test = tr_te.ix[filter_test, :].copy()
        va = tr_te.ix[filter_va, :].copy()
        #
        train = train.drop(['appID','conversionTime', 'clickTime',  'userID', 'day', 'instanceID'], 1)
        test = test.drop(['appID','conversionTime', 'clickTime', 'userID', 'day'], 1)  # 删除train3.csv中的一些列
        va = va.drop(['appID','conversionTime', 'clickTime',  'userID', 'day', 'instanceID'], 1)

        print "--" * 20 + "正在写入文件！" + "--" * 20
        train.fillna(0)
        test.fillna(0)
        va.fillna(0)

        train.to_csv(r'E:\dataset\train_25_1.csv', index=False, sep=',')
        test.to_csv(r'E:\dataset\test_25_1.csv', index=False, sep=',')
        va.to_csv(r'E:\dataset\va_25_1.csv', index=False, sep=',')
        end = time.clock()

        print('Running time: %s Seconds' % (end - start))
        return 1

    #测试新特征效果
    if(0):
        # target = train3['label']
        # data  = train3.drop(['label','conversionTime','day','userID','clickTime'], 1)
        # gbdt = GradientBoostingClassifier()
        # gbdt.fit(data, target)
        #
        # features_importances = pd.DataFrame(gbdt.feature_importances_)
        # # test_features = SelectFromModel(GradientBoostingClassifier()).fit_transform(data, target)
        # data = pd.DataFrame(data)
        # print data.info()
        #
        # features_importances.to_csv(r'E:\dataset\test_features.csv', index=False, sep=',')
        #
        # return 1

        # tr = train3.iloc[:, [0, -1]]
        se = 'train_cluster'

        tr = train3[['label',se]]
        tr1 = DataFrame(tr)
        # tr1['ft'] = train3['train_cluster']
        # tr1 = tr1.sort(columns='train_cluster')
        # tr2 = tr1.groupby(['train_cluster']).sum()
        # tr3 = tr1.train_cluster.value_counts()
        ######################
        tr1 = tr1.sort(columns=se)
        tr2 = tr1.groupby([se]).sum()
        tr3 = tr1[se].value_counts()   #统计每个值出现的次数

        tr4 = pd.concat([tr2, tr3], axis=1)

        tr4.to_csv('feature_test.csv', index=False, sep=',')

        return 1
    if(0):

        # y = test3.creativeID.value_counts()
        # y = pd.DataFrame(y)
        # print y
        # y.to_csv('ytr.csv',index=True, sep=',')
        # return 1

        y1 = pd.merge(train3, y, on='creativeID', how="left")
        y1['count'] = y1['count'] / 500


        y2 = pd.merge(test3, ytr, on='creativeID', how="left")
        y2['count'] = y2['count'] / 500
        train3 = y1
        test3 = y2

    filter_test = np.logical_and(tr_te.day.values > 30, 1)
    filter_train = np.logical_and(tr_te.day.values < 31, tr_te.day.values > 16)
    #训练集

    train = tr_te.ix[filter_train, :].copy()
    train = train.drop(['conversionTime', 'clickTime', 'creativeID', 'userID', 'day','instanceID'], 1)

    #测试集
    test = tr_te.ix[filter_test, :].copy()
    test = test.drop(['conversionTime','clickTime','creativeID','userID','day'], 1)  # 删除train3.csv中的一些列

    print "--"*20+"正在写入文件！"+"--"*20
    train.fillna(0)
    test.fillna(0)

    train.to_csv(r'E:\dataset\train_25_1.csv', index=False, sep=',')
    test.to_csv(r'E:\dataset\test_25_1.csv', index=False, sep=',')
    end = time.clock()

    print('Running time: %s Seconds' % (end - start))
    return 1

    ##############################################################

    #############################################################
    #输入train和test原始的表格，返回合并后的表格
def get_concat(tr,test):
    tr.insert(0, 'instanceID', tr.label)
    tr.instanceID = 0
    tr.conversionTime = 0

    test.insert(3, 'conversionTime', test.clickTime)
    test.conversionTime = 0
    tr_te = pd.concat([tr, test])
    return tr_te

def deal_rate(dateset, deal_data, deal_name):
    tr2 = pd.merge(dateset, deal_data, on=deal_name, how="left")
    tr2 = tr2.fillna(0)
    return tr2

def get_count(data,cu):     #统计cu出现的次数，返回一个对应的列表
    se = cu

    tr = data[['label', se]]
    tr1 = DataFrame(tr)
    ######################
    tr1 = tr1.sort(columns=se)
    tr2 = tr1.groupby([se]).sum()
    tr3 = tr1[se].value_counts()  # 统计每个值出现的次数
    tr4 = pd.concat([tr2, tr3], axis=1)

    tr4['rate'] = tr4.label.values.astype('float') / tr4[cu].values
    tr4['id'] = tr4.index
    tr4 = tr4.drop(['label',cu],1)

    tr4.columns = [cu+'_rate', cu]

    return tr4
    # tr4.to_csv('feature_test.csv', index=False, sep=',')

def ensure_sequence(tr,te):
    tr = tr.sort(columns='userID')
    te = te.sort(columns='instanceID')
    return tr,te
def get_agg(group_by, value, func):
    g1 = pd.Series(value).groupby(group_by)
    agg1  = g1.aggregate(func)
    #print agg1
    r1 = agg1[group_by].values
    return r1

def test_kmeans_cluster(dataset):
    import numpy as np
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    from sklearn.cluster import KMeans
    from sklearn import datasets

    np.random.seed(100)

    X = dataset
    # X =  dataset.loc[:,['adID','camgaignID','advertiserID']]
    # X1 = dataset.loc[:, ['adID', 'creativeID']]
    # X2 = dataset.loc[:, ['camgaignID', 'creativeID']]
    # X3 = dataset.loc[:, ['creativeID','advertiserID']]

    # y = iris.target
    est = KMeans(n_clusters = 30, n_init=10, init='k-means++')
    id_labels = pd.DataFrame()
    try:
        est.fit(X)
    except:
        X = pd.DataFrame(X)
        print X.head(5)
        return 1

    id_labels['l1'] = est.labels_
    # est.fit(X1)
    # id_labels['l2'] = est.labels_
    # est.fit(X2)
    # id_labels['l3'] = est.labels_
    # est.fit(X3)
    # id_labels['l4'] = est.labels_

    return id_labels
    estimators = {
        # 'k_means_iris_3': KMeans(n_clusters=3),
        # 'k_means_iris_8': KMeans(n_clusters=8, n_init=10, init='k-means++')
        # 'k_means_iris_bad_init': KMeans(n_clusters=3, n_init=1,
        #                                 init='random')
    }

    # fignum = 1
    for name, est in estimators.items():
        # fig = plt.figure(fignum, figsize=(4, 3))
        # plt.clf()
        # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        # plt.cla()
        est.fit(X)
        labels = est.labels_

        # ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))

        # ax.w_xaxis.set_ticklabels([])
        # ax.w_yaxis.set_ticklabels([])
        # ax.w_zaxis.set_ticklabels([])
        # ax.set_xlabel('Petal width')
        # ax.set_ylabel('Sepal length')
        # ax.set_zlabel('Petal length')
        # ax.set_title(name)
        # fignum = fignum + 1

        # Plot the ground truth
        # fig = plt.figure(fignum, figsize=(4, 3))
        # plt.clf()
        # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        # plt.cla()

        # for name, label in [('Setosa', 0),
        #                     ('Versicolour', 1),
        #                     ('Virginica', 2)]:
        #     ax.text3D(X[y == label, 3].mean(),
        #               X[y == label, 0].mean() + 1.5,
        #               X[y == label, 2].mean(), name,
        #               horizontalalignment='center',
        #               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        # # Reorder the labels to have colors matching the cluster results
        # y = np.choose(y, [1, 2, 0]).astype(np.float)
        # ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y)
        #
        # ax.w_xaxis.set_ticklabels([])
        # ax.w_yaxis.set_ticklabels([])
        # ax.w_zaxis.set_ticklabels([])
        # ax.set_xlabel('Petal width')
        # ax.set_ylabel('Sepal length')
        # ax.set_zlabel('Petal length')
        # ax.set_title('groud truth')
        # plt.show()
def deal_ui(user_info, tr):
    tr2 = pd.merge(tr, user_info, on='userID', how = "left")
    tr2 = tr2.fillna(0)
    return tr2

def change_seq(train):
    mid = train['userID']
    train.drop(labels=['userID'], axis=1, inplace=True)
    train.insert(6, 'userID', mid)
    return train

def deal_ad(ad,app_categories):

    # print app_categories.info()
    # print app_categories.appCategory.value_counts()
    # print app_categories.describe()   #几种特征值的计算
    # print app_categories.appID.describe()  #输出某一列的统计数据

    new_ad = pd.merge(ad, app_categories, on = 'appID')     #将app的类别合并到ad文件中去
    # new_ad = new_ad.drop('appID', 1)                            #删除冗余的appID列
    return new_ad

def deal_ad1(train, new_ad):
    train1 = pd.merge(train, new_ad, on = 'creativeID')
    # train1 = train1.drop('creativeID', 1)
    return train1

def deal_userapp(ui, ua, appcate):
    # print "正在处理的ui2文件！"  #因为文件过大，只处理一次，后面的就注释掉
    # ui2 = pd.merge(ui, appcate, on = 'appID')
    print "正在处理ua2文件！"
    ua2 = pd.merge(ua, appcate, on = 'appID')
    # ua2 = ua2.drop('appID',1)
    #ui3 = pd.merge(ui, appcate, on = 'appID')
    # return (ui2, ua2)
    return  ua2


def deal_user_info(user_info, tr):
    tr2 = pd.merge(tr, user_info, on = 'userID')
    # tr2 = tr2.fillna(0)
    return tr2

def deal_position(position, tr):
    tr3 = pd.merge(tr, position, on = 'positionID')
    tr3 = tr3.drop('positionID', 1)
    return tr3

if __name__ == '__main__':
    pre_deal()

