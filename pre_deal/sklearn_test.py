# coding=gbk  
''''' 
Created on 2016年6月4日 
 
@author: bryan 
'''  
   
import time    
from sklearn import metrics    
import pickle as pickle    
import numpy as np
import pandas as pd  
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn import datasets
  
    
# Multinomial Naive Bayes Classifier    
def naive_bayes_classifier(train_x, train_y):    
    from sklearn.naive_bayes import MultinomialNB    
    model = MultinomialNB(alpha=100)  
    #from sklearn.naive_bayes import GaussianNB
    #model = GaussianNB(priors=None)  
    model.fit(train_x, train_y)    
    return model    
    
    
# KNN Classifier    
def knn_classifier(train_x, train_y):    
    from sklearn.neighbors import KNeighborsClassifier    
    model = KNeighborsClassifier(n_neighbors=5)    
    model.fit(train_x, train_y)    
    return model    
    
    
# Logistic Regression Classifier    
def logistic_regression_classifier(train_x, train_y):    
    from sklearn.linear_model import LogisticRegression    
    model = LogisticRegression(penalty='l2')    
    model.fit(train_x, train_y)    
    return model    
    
    
# Random Forest Classifier    
def random_forest_classifier(train_x, train_y):    
    from sklearn.ensemble import RandomForestClassifier    
    model = RandomForestClassifier(n_estimators=80)
    model.fit(train_x, train_y)    
    return model    
    
    
# Decision Tree Classifier    
def decision_tree_classifier(train_x, train_y):    
    from sklearn.tree import DecisionTreeClassifier   
    model = DecisionTreeClassifier()    
    model.fit(train_x, train_y)    
    return model    
    
    
# GBDT(Gradient Boosting Decision Tree) Classifier    
def gradient_boosting_classifier(train_x, train_y):    
    from sklearn.ensemble import GradientBoostingClassifier    
    model = GradientBoostingClassifier(n_estimators=200)    
    model.fit(train_x, train_y)    
    return model    
    
    
# SVM Classifier    
def svm_classifier(train_x, train_y):    
    from sklearn.svm import SVC    
    model = SVC(kernel='rbf', probability=True)    
    model.fit(train_x, train_y)    
    return model    
    
# SVM Classifier using cross validation    
def svm_cross_validation(train_x, train_y):    
    from sklearn.grid_search import GridSearchCV    
    from sklearn.svm import SVC    
    model = SVC(kernel='rbf', probability=True)    
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000], 'gamma': [0.001, 0.0001]}    
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)    
    grid_search.fit(train_x, train_y)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
    model.fit(train_x, train_y)    
    return model    
    
#Definition of COLs:
#1. sepal length in cm (花萼长)
#2. sepal width in cm（花萼宽）
#3. petal length in cm (花瓣长)
#4. petal width in cm（花瓣宽）
#5. class: 
#      -- Iris Setosa
#      -- Iris Versicolour
#      -- Iris Virginica
#Missing Attribute Values: None
def read_data(data_name='iris'):
    if data_name == 'iris': 
        # 使用鸢尾花的例子
        ds = datasets.load_iris()
    elif data_name == 'digits':
        # 数字识别的例子
        ds = datasets.load_digits()
    elif data_name == 'diabetes':
        # 糖尿病的例子
        ds = datasets.load_diabetes()
        
    
    # Break up the dataset into non-overlapping training (75%) and testing
    # (25%) sets.
    skf = StratifiedKFold(n_splits=4)
    # Only take the first fold.
    train_index, test_index = next(iter(skf.split(ds.data, ds.target)))
    
    
    X_train = ds.data[train_index]
    y_train = ds.target[train_index]
    X_test = ds.data[test_index]
    y_test = ds.target[test_index]
    
    #print iris.data, iris.target

    return X_train, y_train, X_test, y_test, ds


def test_classifierMethod(data_name):
    #data_file = "H:\\Research\\data\\trainCG.csv"    
    thresh = 0.5    
    model_save_file = None    
    model_save = {}    
     
    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM','SVMCV', 'GBDT']    
    classifiers = { 'NB':naive_bayes_classifier,     
                    'KNN':knn_classifier,    
                    'LR':logistic_regression_classifier,   
                    'RF':random_forest_classifier,    
                   'DT':decision_tree_classifier,    
                   'SVM':svm_classifier,    
                   'SVMCV':svm_cross_validation,    
                   'GBDT':gradient_boosting_classifier    
    }    
    
    print('reading training and testing data...')    
    train_x, train_y, test_x, test_y, ds = read_data(data_name)  
    
    for classifier in test_classifiers:    
        print('******************* %s ********************' % classifier)    
        start_time = time.time()    
        model = classifiers[classifier](train_x, train_y)    
        print('training took %fs!' % (time.time() - start_time))    
        predict = model.predict(test_x)    
        if model_save_file != None:    
            model_save[classifier] = model
        # 正确率 = 提取出的正确信息条数 /  提取出的信息条数    
        # P = TP/(TP+FP) 
        precision = metrics.precision_score(test_y, predict, average='macro')
        # 召回率 = 提取出的正确信息条数 /  样本中的信息条数
        # R = TP/(TP+FN) = 1 - FN/T
        recall = metrics.recall_score(test_y, predict, average='macro')
        print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        # A = (TP + TN)/(P+N)  
        accuracy = metrics.accuracy_score(test_y, predict)    
        print('accuracy: %.2f%%' % (100 * accuracy))
        if data_name == 'digits':
            plt.title('prediction accuracy of %s'%(classifier))
            images_and_labels = list(zip(ds.images, ds.target))
            for index, (image, label) in enumerate(images_and_labels[:10]):
                plt.subplot(5, 5, index + 1)
                plt.axis('off')
                plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
                plt.title('Training: %i' % label)
                
            images_and_predictions = list(zip(ds.images, predict))
            for index, (image, prediction) in enumerate(images_and_predictions[:10]):
                plt.subplot(5, 5, index + 16)
                plt.axis('off')
                plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
                plt.title('Prediction: %i' % prediction)
            plt.show()
    
    if model_save_file != None:    
        pickle.dump(model_save, open(model_save_file, 'wb'))   

###################################################################################################
#
#        
def linear_regresser(train_x, train_y):
    from sklearn import linear_model
    model = linear_model.LinearRegression() 
    model.fit(train_x, train_y)    
    return model

def ridge_regresser(train_x, train_y):
    from sklearn.linear_model import Ridge  
    model = Ridge(alpha=.9)
    model.fit(train_x, train_y) 
    return model

def logistics_regresser(train_x, train_y):
    from sklearn.linear_model import LogisticRegression    
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y) 
    return model

def test_regressionMethod(data_name):
    test_regressers = ['LineReg', 'RidgeReg', 'LogisticReg']    
    regressers = { 'LineReg':linear_regresser,   
                  'RidgeReg': ridge_regresser,  
                  'LogisticReg':logistics_regresser
    }    
    
    print('reading training and testing data...')    
    train_x, train_y, test_x, test_y, ds = read_data(data_name)  
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    train_x_r = pca.fit(train_x).transform(train_x)
    test_x_r = pca.fit(test_x).transform(test_x)
    
    for regresser in test_regressers:    
        print('******************* %s ********************' % regresser)    
        start_time = time.time()    
        model = regressers[regresser](train_x_r, train_y)    
        print('training took %fs!' % (time.time() - start_time))    
         

        # The coefficients
        print('Coefficients: \n', model.coef_)
        # The mean squared error
        print("Mean squared error: %.2f"
              % np.mean((model.predict(test_x_r) - test_y) ** 2))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % model.score(test_x_r, test_y))

        test_x_r.sort()
        predict = model.predict(test_x_r)
        plt.scatter(test_x_r, test_y,  color='black')
        plt.scatter(test_x_r, predict, color='blue')
        
        plt.xticks(())
        plt.yticks(())
        
        plt.show()
        

###################################################################################################
#
#        
def test_kmeans_cluster():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    
    from sklearn.cluster import KMeans
    from sklearn import datasets
    
    np.random.seed(5)
    
    centers = [[1, 1], [-1, -1], [1, -1]]
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    estimators = {'k_means_iris_3': KMeans(n_clusters=3),
                  'k_means_iris_8': KMeans(n_clusters=8),
                  'k_means_iris_bad_init': KMeans(n_clusters=3, n_init=1,
                                                  init='random')}
    
    
    fignum = 1
    for name, est in estimators.items():
        fig = plt.figure(fignum, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    
        plt.cla()
        est.fit(X)
        labels = est.labels_
    
        ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))
    
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')
        ax.set_title(name)
        fignum = fignum + 1
    
    # Plot the ground truth
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    
    plt.cla()
    
    for name, label in [('Setosa', 0),
                        ('Versicolour', 1),
                        ('Virginica', 2)]:
        ax.text3D(X[y == label, 3].mean(),
                  X[y == label, 0].mean() + 1.5,
                  X[y == label, 2].mean(), name,
                  horizontalalignment = 'center',
                  bbox = dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 2, 0]).astype(np.float)
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y)
    
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title('groud truth')
    plt.show()
       
def test_spectualCluster():
    import numpy as np
    import matplotlib.pyplot as plt
    
    from sklearn.feature_extraction import image
    from sklearn.cluster import spectral_clustering
    l = 100
    x, y = np.indices((l, l))
    
    center1 = (28, 24)
    center2 = (40, 50)
    center3 = (67, 58)
    center4 = (24, 70)
    
    radius1, radius2, radius3, radius4 = 16, 14, 15, 14
    
    circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
    circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
    circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2
    circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4 ** 2
    
    img = circle1 + circle2 + circle3 + circle4
    
    # We use a mask that limits to the foreground: the problem that we are
    # interested in here is not separating the objects from the background,
    # but separating them one from the other.
    mask = img.astype(bool)
    
    img = img.astype(float)
    img += 1 + 0.2 * np.random.randn(*img.shape)
    
    # Convert the image into a graph with the value of the gradient on the
    # edges.
    graph = image.img_to_graph(img, mask=mask)
    
    # Take a decreasing function of the gradient: we take it weakly
    # dependent from the gradient the segmentation is close to a voronoi
    graph.data = np.exp(-graph.data / graph.data.std())
    
    # Force the solver to be arpack, since amg is numerically
    # unstable on this example
    labels = spectral_clustering(graph, n_clusters=4, eigen_solver='arpack')
    label_im = -np.ones(mask.shape)
    label_im[mask] = labels
    
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    
    ax1.matshow(img)
    ax2.matshow(label_im)
    
    # two circle
    img = circle1 + circle2
    mask = img.astype(bool)
    img = img.astype(float)
    
    img += 1 + 0.2 * np.random.randn(*img.shape)
    
    graph = image.img_to_graph(img, mask=mask)
    graph.data = np.exp(-graph.data / graph.data.std())
    
    labels = spectral_clustering(graph, n_clusters=2, eigen_solver='arpack')
    label_im = -np.ones(mask.shape)
    label_im[mask] = labels
    
    
    ax3.matshow(img)
    ax4.matshow(label_im)
    
    plt.show()
 
 
 
###################################################################################################
#
#        
def test_pca_lda():
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    iris = datasets.load_iris()
    
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)
    
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)
    
    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))
    
    plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2
    
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        ax1.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    ax1.legend(loc='best', shadow=False, scatterpoints=1)
    ax1.set_title('PCA of IRIS dataset')
    
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        ax2.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    ax2.legend(loc='best', shadow=False, scatterpoints=1)
    ax2.set_title('LDA of IRIS dataset')
    plt.show()
    
        
##############################################################################################################################################################
# 汉明距离
distance = lambda a,b : 0 if a==b else 1  
   
def dtw(sa,sb):  
    ''''' 
    >>>dtw(u"干啦今今今今今天天气气气气气好好好好啊啊啊", u"今天天气好好啊") 
    2 
    '''  
    MAX_COST = 1<<32  
    #初始化一个len(sb) 行(i)，len(sa)列(j)的二维矩阵  
    len_sa = len(sa)  
    len_sb = len(sb)  
    # BUG:这样是错误的(浅拷贝): dtw_array = [[MAX_COST]*len(sa)]*len(sb)  
    dtw_array = [[MAX_COST for i in range(len_sa)] for j in range(len_sb)]  
    dtw_array[0][0] = distance(sa[0],sb[0])  
    for i in xrange(0, len_sb):  
        for j in xrange(0, len_sa):  
            if i+j==0:  
                continue  
            nb = []  
            if i > 0: nb.append(dtw_array[i-1][j])  
            if j > 0: nb.append(dtw_array[i][j-1])  
            if i > 0 and j > 0: nb.append(dtw_array[i-1][j-1])  
            min_route = min(nb)  
            cost = distance(sa[j],sb[i])  
            dtw_array[i][j] = cost + min_route  
    return dtw_array[len_sb-1][len_sa-1]  
   
   
def test_distance():  
    s1 = u'干啦今今今今今天天气气气气气好好好好啊啊啊'  
    s2 = u'今天天气好好啊啊啊'  
    d = dtw(s1, s2)  
    print d  
    return 0  

##############################################################################
#
def test_lineregress():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn import datasets, linear_model
    from sklearn.cross_validation import train_test_split
    from sklearn.linear_model import LinearRegression

    ccpp_csv = 'D:\\djpy\\skl_sample\\data\\CCPP\\ccpp.csv'
    data = pd.read_csv(ccpp_csv)
    X = data[['AT', 'V', 'AP', 'RH']]
    y = data[['PE']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    
    #模型拟合测试集
    y_pred = linreg.predict(X_test)
    from sklearn import metrics
    # 用scikit-learn计算MSE
    print "MSE:",metrics.mean_squared_error(y_test, y_pred)
    # 用scikit-learn计算RMSE
    print "RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    
    
    from sklearn.model_selection import cross_val_predict
    predicted = cross_val_predict(linreg, X, y, cv=10)
    # 用scikit-learn计算MSE
    print "MSE:",metrics.mean_squared_error(y, predicted)
    # 用scikit-learn计算RMSE
    print "RMSE:",np.sqrt(metrics.mean_squared_error(y, predicted))
    
    
    fig, ax = plt.subplots()
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

   

if __name__ == '__main__':  
    #test_pca_lda()
    
    # test_regressionMethod('diabetes')
    
    
    # test_classifierMethod('digits')
    #test_classifierMethod('iris')
    
    test_kmeans_cluster()
    #test_spectualCluster()
    
    #test_lineregress()
    #test_distance() 
    