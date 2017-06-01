# coding=utf-8
import pandas as pd
from  pandas import Series, DataFrame
import matplotlib.pyplot as  plt
import time
import numpy as np
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.decomposition import PCA
# start =time.clock()
# end = time.clock()
# print('Running time: %s Seconds'%(end-start))

def pre_deal():
    try:
        # ad = pd.read_csv(r'E:\dataset\ad.csv', sep=',')  # 只能读取英文路径,返回的是一个dataframe类型数据
        # app_cate = pd.read_csv(r'E:\dataset\app_categories_del.csv', sep=',')
        # tr = pd.read_csv(r'E:\dataset\train.csv', sep = ',')
        # position = pd.read_csv(r'E:\dataset\position.csv', sep=',')
        # user_info = pd.read_csv(r'E:\dataset\user.csv', sep=',')
        #
        # user_installedapps = pd.read_csv(r'E:\dataset\user_installedapps.csv', sep = ',')#读取的最大的文件，尽可能避免处理此文件
        # user_installedapps = pd.read_csv(r'E:\dataset\position.csv', sep=',')#假的
        # user_app_actions = pd.read_csv(r'E:\dataset\user_app_actions.csv', sep=',')
        # train = pd.read_csv(r'train_17_1.csv', sep = ',')

        # train = pd.read_csv(r'train_test3.csv', sep=',')
        # test = pd.read_csv(r'test_test3.csv', sep=',')
        # test = pd.read_csv(r'E:\dataset\results\test3.csv', sep=',')
        # np.savetxt('E:\\forpython\\featvector.csv', data_to_save, delimiter=',')
        # ui = pd.read_csv(r'user_install3.csv',sep = ',')
        # app_cate = pd.read_csv(r'E:\dataset\app_categories.csv', sep=',')
        # train = pd.read_csv(r'E:\dataset\results\train.csv', sep=',')
        # test = pd.read_csv(r'E:\dataset\results\test.csv', sep=',')

        # train_id = pd.read_csv(r'E:\dataset\train_ids.csv', sep=',')
        tr_te = pd.read_csv(r'E:\dataset\results\tr_te_27_1.csv', sep = ',')
        label_1 = pd.read_csv(r'E:\dataset\results\label_1.csv', sep=',')

    except:
        print "打开文件的时候出错!" * 10
    #测试新数据的效果及分析用户
    if(0):
        train = np.logical_and(tr_te.label > -1,1)
        train_data = tr_te.ix[train,:]

        # train_data['hometown_fa'] = pd.factorize(train_data['hometown'])[0]
        # train_data['residence_fa'] = pd.factorize(train_data['residence'])[0]
        train_data['residence_hometown'] = train_data['residence'] + train_data['hometown']
        train_data['site_position'] = train_data['sitesetID'] + train_data['positionType']
        test_features = ['connectionType','telecomsOperator',
                         'appPlatform','appCategory','gender','education',
                         'marriageStatus','haveBaby','sitesetID','positionType']

        test_features_new = ['residence_hometown','site_position']

        for features in test_features_new:
            means = train_data.groupby([train_data[features], train_data['label']])['label']
            means1 = train_data.groupby([train_data[features]])[features]
            print features,"的特征差异"
            print means.size() / means1.size()

        label_1['converse_time_day'] = (label_1.conversionTime.values / 10000) - (label_1.clickTime.values / 10000)
        label_1['converse_time_hour'] = (label_1.conversionTime.values / 100 % 100)-(label_1.clickTime.values / 100 % 100)
        label_1['converse_time_mintue'] = (label_1.conversionTime.values % 100) -(label_1.clickTime.values % 100)
        label_1['converse_time'] = 24*60*label_1['converse_time_day'] +60*label_1['converse_time_hour']+label_1['converse_time_mintue']
        # df['sex_age'] = df['age'] + df['sex'] * 100

        #可视化
        # plt.hist(label_1.converse_time_hour)
        # plt.hist(label_1.converse_time_hour, bins=8, color=sns.desaturate("indianred", .8), alpha=.4)
        # sns.factorplot(x="age", y="label", hue='gender', data=tr_te, kind="bar", size=7, ci=None, aspect=1.5)
        # plt.show()
        return 1


    if(0):
        print tr_te.appCategory.value_counts()
        tr = pd.merge(tr_te, ui, on='userID', how="left")
        tr = tr.fillna(0)
        tr['ui'] = 0
        li = tr.appCategory.values
        li.astype('str')
        tr.ix[tr[li] == 1, 'ui'] = 1  # 按一列的值的条件 给另一列赋值
        # print tr.head(10)
        return 1

    sample = 0
    if(sample):                            #均匀采样10%的数据作为训练集
        sample_pct = 0.1
        if sample_pct < 1.0:
            np.random.seed(999)
            filter_train = np.logical_and((tr_te.clickTime.values /10000)  < 31, (tr_te.clickTime.values / 10000) > 16)
            train = tr_te.ix[filter_train, :].copy()
            r1 = np.random.uniform(0, 1, train.shape[0])  # 功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
            train = train.ix[r1 < sample_pct, :]
            print "testing with small sample of training data, ", train.shape  # 采样的验证集
    tr_te['day'] = deal_date(tr_te.clickTime.values)

    tr_te['residence_hometown'] = tr_te['residence'] + tr_te['hometown']
    tr_te['site_position'] = tr_te['sitesetID'] + tr_te['positionType']

    count_list = ['camgaignID','appID','adID', 'creativeID', 'advertiserID', 'positionID', 'residence', 'hometown',
                  ]

    n_ks = {'camgaignID':100,'appID':100,'adID': 100, 'creativeID': 100, 'advertiserID': 100, 'positionID': 50, 'residence': 50,
            'hometown': 100}
    exp2_dict = {}
    for vn in count_list:
        exp2_dict[vn] = np.zeros(tr_te.shape[0])
    days_npa = tr_te.day.values

    #历史转换率的统计
    for day_v in xrange(18, 32):
        df1 = tr_te.ix[np.logical_and(tr_te.day.values < day_v, tr_te.day.values < 31), :].copy()
        df2 = tr_te.ix[tr_te.day.values == day_v, :].copy()
        pred_prev = df1.label.values.mean() * np.ones(df1.shape[0])  # 历史点击率的平均值

        for vn in count_list:  # 删掉什么
            if 'exp2_' + vn in df1.columns:
                df1.drop('exp2_' + vn, inplace=True, axis=1)

        pred1 = df1.label.values.mean()

        for i in xrange(3):
            for vn in count_list:
                p1 = calcLeaveOneOut2(df1, vn, 'label', n_ks[vn], 0, 0.25, mean0=pred_prev)
                pred = pred_prev * p1
                print day_v, i, vn, "change = ", ((pred - pred_prev) ** 2).mean()
                pred_prev = pred

            pred1 = df1.label.values.mean()

            for vn in count_list:
                print "=" * 20, "merge", day_v, vn
                diff1 = mergeLeaveOneOut2(df1, df2, vn)
                pred1 *= diff1
                exp2_dict[vn][days_npa == day_v] = diff1

            pred1 *= df1.label.values.mean() / pred1.mean()


    for vn in count_list:
        tr_te['exp2_' + vn] = exp2_dict[vn]

    tr_te['residence_hometown_model'] = np.add(tr_te.hometown.values, tr_te.residence.values)





    print "Finish load data,",tr_te.shape

    print tr_te.head(10)


    start = time.clock()

    concat = 0
    if(concat):
        tr_te = get_concat(train, test)
        tr_te1.sort(['instanceID','clickTime'])
        tr_te.sort(['instanceID', 'clickTime'])
        tr_te.appCategory = tr_te1.appCategory

        tr_te.to_csv(r'E:\dataset\results\tr_te_27_1.csv', index=False, sep=',')
        print tr_te.appCategory.value_counts()
        return 1
    #时间处理310000

    # tr_te['hour1'] = np.round(tr_te.clickTime / 100 % 100)  # 每天的小时
    # tr_te['day_hour'] = (tr_te.day.values - 17) * 24 + tr_te.hour1.values  # 时间信息转化成为小时表达
    # tr_te['day_hour_prev'] = tr_te['day_hour'] - 1  # 每个数据前一小时的值
    # tr_te['day_hour_next'] = tr_te['day_hour'] + 1  # 后一小时值
    #统计点击次数
    # list_click_count = ['appID','adID','creativeID','advertiserID','positionID']
    # for list_ID in list_click_count:
    #     tr_te['cnt_click_'+ list_ID] = get_agg(tr_te_count[list_ID].values, tr_te_count.index, np.size)

    #统计历史转换率
    # list = ['adID','creativeID','advertiserID','positionID','residence','hometown']
    # list = []

    if (1):
        filter_test = np.logical_and(tr_te.day.values == 31, 1)
        filter_train = np.logical_and(tr_te.day.values < 23, tr_te.day.values > 16)
        # filter_va = np.logical_and(tr_te.day.values < 31, tr_te.day.values > 29)
        # 训练集,测试集
        train = tr_te.ix[filter_train, :].copy()
        test = tr_te.ix[filter_test, :].copy()

        # va = tr_te.ix[filter_va, :].copy()
        #'adID','creativeID','advertiserID','positionID','appID',
        train = train.drop(['creativeID','appID','conversionTime', 'clickTime','conversionTime', 'userID', 'day', 'instanceID'], 1)
        # va = va.drop(['adID','creativeID','advertiserID','positionID','appID', 'conversionTime', 'clickTime','conversionTime', 'userID', 'day', 'instanceID'], 1)

        test = test.drop(['creativeID','appID','conversionTime', 'clickTime','conversionTime', 'userID', 'day'], 1)  # 删除train3.csv中的一些列
        '''
        train = train.drop(['appID', 'conversionTime', 'clickTime', 'userID', 'day', 'instanceID',
                            'cluster_positionType_connectionType', 'cluster_advertiserID_positionType',
                            'cluster_gender_positionType', 'cluster_hometown_residence', 'cluster_age_marriageStatus',
                            'cluster_age_positionType', 'cluster_appCategory_positionType', 'cluster_hometown_positionType',
                            'cluster_positionType_telecomsOperator', 'cluster_creativeID_positionType',
                            'cluster_education_positionType', 'cluster_camgaignID_positionType', 'cluster_age_creativeID',
                            'cluster_adID_gender', 'cluster_creativeID_gender', 'cluster_camgaignID_gender',
                            'cluster_appID_gender', 'cluster_marriageStatus_positionType', 'cluster_residence_positionType',
                            'cluster_age_telecomsOperator'], 1)

        test = test.drop(['appID', 'conversionTime', 'clickTime', 'userID', 'day',
                          'cluster_positionType_connectionType', 'cluster_advertiserID_positionType',
                          'cluster_gender_positionType', 'cluster_hometown_residence', 'cluster_age_marriageStatus',
                          'cluster_age_positionType', 'cluster_appCategory_positionType',
                          'cluster_hometown_positionType',
                          'cluster_positionType_telecomsOperator', 'cluster_creativeID_positionType',
                          'cluster_education_positionType', 'cluster_camgaignID_positionType','cluster_age_creativeID',
                          'cluster_adID_gender', 'cluster_creativeID_gender', 'cluster_camgaignID_gender',
                          'cluster_appID_gender', 'cluster_marriageStatus_positionType',
                          'cluster_residence_positionType',
                          'cluster_age_telecomsOperator'
                          ], 1)  # 删除train3.csv中的一些列
        '''
        print "--" * 20 + "正在写入文件！" + "--" * 20
        # train.fillna(0)
        # test.fillna(0)

        train.to_csv(r'E:\dataset\train_31_1.csv', index=False, sep=',')
        test.to_csv(r'E:\dataset\test_31_1.csv', index=False, sep=',')
        # va.to_csv(r'E:\dataset\va_27_1.csv', index=False, sep=',')
        end = time.clock()

        print('Running time: %s Seconds' % (end - start))
        return 1

    print test.adID.value_counts()

    test['cnt_ad_id'] = get_agg(test.adID.values, test.instanceID, np.size)#统计每一个ID出现的次数，并归一化处理
    test['ad_id_cnt2'] = np.minimum(test.cnt_ad_id.astype('int32').values, 300)#出现次数大于三百的用300表示
    test['ad_id2plus'] = test.adID.values                                      #方便对比每个ID出现的次数
    test.ix[test.cnt_ad_id.values == 1, 'ad_id2plus'] = '___only1'
    test.to_csv(r'E:\dataset\test_count.csv', sep=',')
    return 1

    ############################################################
    #将ID数据进行聚类利用creativeID,adID,camgaignID,advertiserID
    if(0):
        train_id = train.loc[:,['appCategory','clickTime']]
        test_id = test.loc[:, ['appCategory','clickTime']]
        # train_id.to_csv(r'E:\dataset\train_s.csv', index = True, sep = ',')
        # test_id.to_csv(r'E:\dataset\test_ids.csv', index = True, sep = ',')

        start = time.clock()
        train_ids = test_kmeans_cluster(train_id)
        test_ids = test_kmeans_cluster(test_id)
        print "*********************train_ids******************"
        print train_id.head(10)
        print "*********************test_ids******************"
        print test_id.head(10)
        print "正在写入文件！！！！！"
        train_ids.to_csv(r'E:\dataset\train_time_cluster.csv',index = False, sep = ',')
        test_ids.to_csv(r'E:\dataset\test_time_cluster.csv', index=False, sep =',')
        end = time.clock()
        print('Running time: %s Seconds'%(end-start))
        return 1
    #仅利用地址信息的一级编码hometown,residence
    '''
    user_residence = user_info.residence / 100
    user_hometown = user_info.hometown / 100
    user_residence = user_residence.astype('int')
    user_hometown = user_hometown.astype('int')

    user_info['hometown'] = user_hometown
    user_info['residence'] = user_residence
    user_info.to_csv(r'E:\dataset\user_del_place.csv', index = False, sep = ',')
    print user_info.hometown.value_counts()
    print('******************* hometown的类型********************')
    print user_info.residence.value_counts()
    print('******************* residence的类型********************')
    # print app_cate.appCategory.value_counts()
    # print app_cate1.appCategory.value_counts()
    return 1
    '''
    # start = time.clock()
    # lables = [0,1,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    # train['age'] = pd.cut(train.age, [0,1,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,47,61,100], right=False, labels=lables)
    # test['age'] = pd.cut(test.age, [0,1,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,47,61,100], right=False, labels=lables)
    #
    # train.fillna(24.0)
    # test.fillna(24.0)
    # # print train.appCategory.value_counts()
    #
    # a1 = OneHotEncoder(sparse=False).fit_transform(app_cate[['appCategory']])
    # a2 = app_cate.drop(['appCategory'],1)
    # # enc = preprocessing.OneHotEncoder()
    # # ad_categary = app_cate.appCategory
    # # ad_categary = enc.fit(ad_categary).transform(ad_categary)
    # a1 = DataFrame(a1)
    # # enc.transform([[0, 1, 3]]).toarray()
    # a3 = pd.concat([a2, a1], axis=1)
    # a3.to_csv(r'E:\dataset\app_cate_onehot.csv', index = False, sep = ',')
    # print a3.head(10)

    '''
    print train.adID.value_counts()
    print train.camgaignID.value_counts()
    print train.advertiserID.value_counts()
    print train.residence.value_counts()
    '''

    #########################PCA降维算法，利用SVD的方法
    if (False):
        label = train.iloc[:, 0:17]
        instance_label = test.iloc[:,0:18]

        train = train.drop(['label','clickTime','connectionType','telecomsOperator','camgaignID','advertiserID','appPlatform','appCategory','age','gender','education','marriageStatus','haveBaby','hometown','residence','sitesetID','positionType'],1)
        test = test.drop(['instanceID', 'label','clickTime','connectionType','telecomsOperator','camgaignID','advertiserID','appPlatform','appCategory','age','gender','education','marriageStatus','haveBaby','hometown','residence','sitesetID','positionType'],1)

        pca = PCA(n_components=10)
        train_pca = pca.fit(train).transform(train)
        test_pca = pca.fit(test).transform(test)

        train_pca = DataFrame(train_pca)
        test_pca = DataFrame(test_pca )

        train_pca = pd.concat([label, train_pca], axis =1)
        test_pca = pd.concat([instance_label, test_pca], axis=1)

        train_pca.to_csv(r'E:\dataset\train_pca_19_2.csv', index=False, sep=',')
        test_pca.to_csv(r'E:\dataset\test_pca_19_2.csv', index=False, sep=',')
        return 1
    #######################################################################
    # print ui.head(10)
    # ui = ui.drop('appID',1)#排序，删除appID，去除重复的用户
    # ui.duplicated()
    # print ui.head(10)
    #
    # ui.sort_value(columns = 'userID')
    #
    # print ui.head(100)
    ######################################################################
    #########################################尝试提取时间信息
    '''
    train1 = train.clickTime/100
    train1 = train1%100
    # train1 = DataFrame(train1)
    train1 = train1.astype('int')
    train['clickTime'] = train1
    train.to_csv(r'E:\dataset\train_time.csv', index=False, sep=',')
    '''
    # train2 = train.clickTime%10000
    # train2 = train2/100
    #
    # train1 = DataFrame(train1)
    # train2 = DataFrame(train2)

    # train1 = train1['clickTime'].astype('int')
    # train2 = train2['clickTime'].astype('int')



    # print train1.clickTime.value_counts()
    # print train2.clickTime.value_counts()

    # print train1.head(20)
    # print train2.tail(20)

    # user_pca.to_csv(r'E:\dataset\user_pca.csv', index=False, sep=',')
    # test_x_r = pca.fit(test_x).transform(test_x)

    # train = train.drop(['creativeID'],1)
    # train.to_csv(r'E:\dataset\results\train4.csv', index=False, sep=',')
    # #
    # test = test.drop(['creativeID'], 1)
    # test.to_csv(r'E:\dataset\results\test4.csv', index=False, sep=',')
    #
    # end = time.clock()
    # print('Running time: %s Seconds' % (end - start))
    #第一个参数是需要统计的


def calcLeaveOneOut2(df, vn, vn_y, cred_k, r_k, power, mean0=None, add_count=False):
    if mean0 is None:
        mean0 = df[vn_y].mean() * np.ones(df.shape[0])

    _key_codes = df[vn].values
    grp1 = df[vn_y].groupby(_key_codes)
    grp_mean = pd.Series(mean0).groupby(_key_codes)
    mean1 = grp_mean.aggregate(np.mean)
    sum1 = grp1.aggregate(np.sum)
    cnt1 = grp1.aggregate(np.size)

    # print sum1
    # print cnt1
    vn_sum = 'sum_' + vn
    vn_cnt = 'cnt_' + vn
    _sum = sum1[_key_codes].values
    _cnt = cnt1[_key_codes].values
    _mean = mean1[_key_codes].values
    # print _sum[:10]
    # print _cnt[:10]
    # print _mean[:10]
    # print _cnt[:10]
    _mean[np.isnan(_sum)] = mean0.mean()
    _cnt[np.isnan(_sum)] = 0
    _sum[np.isnan(_sum)] = 0
    # print _cnt[:10]
    _sum -= df[vn_y].values
    _cnt -= 1
    # print _cnt[:10]
    vn_yexp = 'exp2_' + vn
    #    df[vn_yexp] = (_sum + cred_k * mean0)/(_cnt + cred_k)
    diff = np.power((_sum + cred_k * _mean) / (_cnt + cred_k) / _mean, power)
    if vn_yexp in df.columns:
        df[vn_yexp] *= diff
    else:
        df[vn_yexp] = diff
    # if r_k > 0:
    #     df[vn_yexp] *= np.exp((np.random.rand(np.sum(filter_train)) - .5) * r_k)
    if add_count:
        df[vn_cnt] = _cnt
    return diff

def mergeLeaveOneOut2(df, dfv, vn):
    _key_codes = df[vn].values
    vn_yexp = 'exp2_' + vn
    grp1 = df[vn_yexp].groupby(_key_codes)
    _mean1 = grp1.aggregate(np.mean)

    _mean = _mean1[dfv[vn].values].values

    _mean[np.isnan(_mean)] = _mean1.mean()

    return _mean

def get_convser_label(data,attr):

    filter_data = np.logical_and(data.day.values < 24, data.day.values > 16)
    data = data.ix[filter_data, :].copy()
    # print data[attr].value_counts()
    grouped = data['label'].groupby(data[attr])
    t1 = grouped.sum()
    t1 = pd.DataFrame(t1)
    t1[attr] = t1.index
    t1.columns = [attr +'_label', attr]
    t1[0] = t1[0] / data.shape[0]
    return t1

def get_agg(group_by, value, func):
    g1 = pd.Series(value).groupby(group_by)
    agg1  = g1.aggregate(func)
    #print agg1
    r1 = agg1[group_by].values
    return r1

def deal_date(click_date):
    lables = [1.1, 2, 3, 4, 5, 6, 7, 1.2]
    click_date_day = click_date / 10000
    # test3_ = pd.cut(test3_, [0,1, 7, 12, 14, 18, 22,23,24], right=False, labels=lables)
    click_date_day = click_date_day.astype('int')
    # test3 = test3.sort(columns = 'instanceID')
    ##########对train的时间处理
    # train3_ = train3.clickTime / 10000
    # train3_ = pd.cut(train3_, [0,1, 7, 12, 14, 18, 22, 23,24], right=False, labels=lables)
    # train3_ = train3_.astype('int')

    # train3, test3 = ensure_sequence(train3,test3)
    # 给时间区间重新分段
    # train3=train3.fillna(0)
    # test3=test3.fillna(0)
    return click_date_day
def get_concat(tr,test):
    tr.insert(0, 'instanceID', tr.label)
    tr.instanceID = 0
    # tr.conversionTime = 0

    test.insert(3, 'conversionTime', test.clickTime)
    test.conversionTime = 0
    tr_te = pd.concat([tr, test])

    tr_te.fillna(0)
    return tr_te
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

    np.random.seed(5)

    X = dataset
    # X =  dataset.loc[:,['adID','camgaignID','advertiserID']]
    # X1 = dataset.loc[:, ['adID', 'creativeID']]
    # X2 = dataset.loc[:, ['camgaignID', 'creativeID']]
    # X3 = dataset.loc[:, ['creativeID','advertiserID']]

    # y = iris.target
    est = KMeans(n_clusters=10, n_init=10, init='k-means++')
    id_labels = pd.DataFrame()

    est.fit(X)
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

def deal_ad(ad, app_categories):
    # print app_categories.info()
    # print app_categories.appCategory.value_counts()
    # print app_categories.describe()   #几种特征值的计算
    # print app_categories.appID.describe()  #输出某一列的统计数据

    new_ad = pd.merge(ad, app_categories, on='appID')  # 将app的类别合并到ad文件中去
    new_ad = new_ad.drop('appID', 1)  # 删除冗余的appID列
    return new_ad


def deal_ad1(train, new_ad):
    train1 = pd.merge(train, new_ad, on='creativeID')
    train1 = train1.drop('creativeID', 1)
    return train1


def deal_userapp(ui, ua, appcate):
    # print "正在处理的ui2文件！"  #因为文件过大，只处理一次，后面的就注释掉
    # ui2 = pd.merge(ui, appcate, on = 'appID')
    print "正在处理ua2文件！"
    ua2 = pd.merge(ua, appcate, on='appID')
    ua2 = ua2.drop('appID', 1)
    # ui3 = pd.merge(ui, appcate, on = 'appID')
    # return (ui2, ua2)
    return ua2


def deal_user_info(user_info, tr):
    tr2 = pd.merge(tr, user_info, on='userID')
    return tr2


def deal_position(position, tr):
    tr3 = pd.merge(tr, position, on='positionID')
    tr3 = tr3.drop('positionID', 1)
    return tr3


if __name__ == '__main__':
    pre_deal()

