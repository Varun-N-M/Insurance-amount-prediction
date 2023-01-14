# Building a machine learning model to predict the amount spent to the utmost accuracy.
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV

from sklearn.decomposition import PCA

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.cluster import KMeans

from sklearn.feature_selection import RFE

d1 = pd.read_excel('Medibuddy insurance data personal details (1) (2).xlsx')
d2 = pd.read_csv('Medibuddy Insurance Data Price (1) (2).csv')

df = pd.merge(d2,d1).drop('Policy no.',axis=1)

print(df.head().to_string())
print(df.info())
print(df.describe())

def grab_col_names(dataframe, car_th=10, cat_th=20):
    cat_col = [col for col in dataframe.columns if dataframe[col].dtype == 'O']
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtype != 'O']
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > cat_th and dataframe[col].dtype == 'O']

    cat_col = cat_col + num_but_cat
    cat_col = [col for col in cat_col if col not in cat_but_car]

    num_col = [col for col in dataframe.columns if dataframe[col].dtype != 'O']
    num_col = [col for col in num_col if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_col)}')
    print(f'num_cols: {len(num_col)}')

    return cat_col, num_col, cat_but_car
print( grab_col_names(df))

b_plot = df.boxplot(vert = False)

le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])


def replace_outlier (mydf, col, method = 'Quartile',strategy = 'median' ):
    if method == 'Quartile':
        Q1 = mydf[col].quantile (0.25)
        Q2 =  mydf[col].quantile (0.50)
        Q3 = mydf[col].quantile (0.75)
        IQR =Q3 - Q1
        LW = Q1 - 1.5 * IQR
        UW = Q3 + 1.5 * IQR
    elif method == 'standard deviation':
        mean = mydf[col].mean()
        std = mydf[col].std()
        LW = mean - (2*std)
        UW = mean + (2*std)
    else:
        print('Pass a correct method')
#Printing all the outliers
    outliers = mydf.loc[(mydf[col]<LW) | (mydf[col]>UW),col]
    outliers_density = round(len(outliers)/ len(mydf),2)
    if len(outliers)==0:
        print(f'feature {col} does not have any outliers')
    else:
        print(f'feature {col} has outliers')
        # print(f'Total number of outliers in this {col} is:'(len(outliers)))
        print(f'Outliers percentage in {col} is {outliers_density*100}%')
    if strategy=='median':
    #mydf.loc[ (mydf[col] < LW), col] = Q2 # used for first method
    #mydf.loc[ (mydf[col] > UW), col] = Q2 # used for first method
        mydf.loc[(mydf [col] < LW), col] = Q1 # second method.. the data may get currupted. so we are res
        mydf.loc[(mydf [col] > UW), col] = Q3 #second method.. as the outliers are more and not treated
    elif strategy == 'mean':
        mydf.loc[(mydf [col] < LW), col] = mean
        mydf.loc[(mydf [col] > UW), col] = mean
    else:
        print('Pass the correct strategy')
    return mydf

def odt_plots (mydf, col):
    f, (ax1, ax2) = plt.subplots (1,2,figsize=(25, 8))
    # descriptive statistic boxplot
    sns.boxplot (mydf [col], ax = ax1)
    ax1.set_title (col + ' boxplot')
    ax1.set_xlabel('values')
    ax1.set_ylabel('boxplot')
    #replacing the outliers
    mydf_out = replace_outlier (mydf, col)
    #plotting boxplot without outliers
    sns.boxplot (mydf_out[col], ax = ax2)
    ax2.set_title (col + 'boxplot')
    ax2.set_xlabel('values')
    ax2.set_ylabel('boxplot')
    plt.show()

for col in df.drop('charges in INR',axis = 1).columns:
    odt_plots(df,col)
b_plot = df.boxplot(vert = False)
plt.show()
corr = df.corr()
fig = plt.figure(figsize=(10,8))
h_plot = sns.heatmap(corr,annot=True)
# plt.show()

def VIF(independent_variables):
    vif = pd.DataFrame()
    vif['vif'] = [variance_inflation_factor (independent_variables.values,i) for i in range (independent_variables.shape[1])]
    vif['independent_variables']= independent_variables.columns
    vif = vif.sort_values(by=['vif'],ascending=False)      #to sort the values in descending order
    return vif
VIF(df.drop('charges in INR',axis=1))

def CWT (data, tcol):
    independent_variables = data.drop(tcol, axis=1).columns
    corr_result = []
    for col in independent_variables :
        corr_result.append(data[tcol].corr(data[col]))
    result = pd.DataFrame([independent_variables, corr_result], index=['independent variables', 'correlation']).T    #T is for transpose
    return result.sort_values(by = 'correlation',ascending = False)
CWT(df,'charges in INR')


def PCA_1(x):
    n_comp = len(x.columns)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Applying PCA
    for i in range(1, n_comp):
        pca = PCA(n_components=i)
        p_comp = pca.fit_transform(x)
        evr = np.cumsum(pca.explained_variance_ratio_)
        if evr[i - 1] > 0.9:
            n_components = i
            break
    print('Ecxplained varience ratio after pca is: ', evr)
    # creating a pcs dataframe
    col = []
    for j in range(1, n_comp):
        col.append('PC_' + str(j))
    pca_df = pd.DataFrame(p_comp, columns=col)
    return pca_df

transformed_df = PCA_1(df.drop('charges in INR',axis = 1))

transformed_df = transformed_df.join(df['charges in INR'],how = 'left')
transformed_df.head()

def train_and_test_split(data,t_col, testsize=0.3):
    x = data.drop(t_col, axis=1)
    y = data[t_col]
    return train_test_split(x,y,test_size=testsize, random_state=42)

def model_builder(model_name, estimator, data, t_col):
    x_train,x_test,y_train,y_test = train_and_test_split(data, t_col)
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    accuracy = r2_score(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    return [model_name, accuracy, rmse]

def multiple_models(data,t_col):
    col_names = ['model_name', 'r2_score', 'RMSE']
    result = pd.DataFrame(columns=col_names)
    result.loc[len(result)] = model_builder('LinearRegression',LinearRegression(),data,t_col)
    result.loc[len(result)] = model_builder('Lasso',Lasso(),data,t_col)
    result.loc[len(result)] = model_builder('Ridge',Ridge(),data,t_col)
    result.loc[len(result)] = model_builder('DecisionTreeRegressor',DecisionTreeRegressor(),data,t_col)
    result.loc[len(result)] = model_builder('KneighborRegressor',KNeighborsRegressor(),data,t_col)
    result.loc[len(result)] = model_builder('RandomForestRegressor',RandomForestRegressor(),data,t_col)
    result.loc[len(result)] = model_builder('SVR',SVR(),data,t_col)
    result.loc[len(result)] = model_builder('AdaBoostRegressor',AdaBoostRegressor(),data,t_col)
    result.loc[len(result)] = model_builder('GradientBoostingRegressor',GradientBoostingRegressor(),data,t_col)
    result.loc[len(result)] = model_builder('XGBRegressor',XGBRegressor(),data,t_col)
    return result.sort_values(by='r2_score',ascending=False)
print( )
print(multiple_models(transformed_df,'charges in INR'))

def kfoldCV(x, y, fold=10):
    score_lr = cross_val_score(LinearRegression(), x, y, cv=fold)
    score_las = cross_val_score(Lasso(), x, y, cv=fold)
    score_ri = cross_val_score(Ridge(), x, y, cv=fold)
    score_dt = cross_val_score(DecisionTreeRegressor(), x, y, cv=fold)
    score_kn = cross_val_score(KNeighborsRegressor(), x, y, cv=fold)
    score_rf = cross_val_score(RandomForestRegressor(), x, y, cv=fold)
    score_svr = cross_val_score(SVR(), x, y, cv=fold)
    score_ab = cross_val_score(AdaBoostRegressor(), x, y, cv=fold)
    score_gb = cross_val_score(GradientBoostingRegressor(), x, y, cv=fold)
    score_xb = cross_val_score(XGBRegressor(), x, y, cv=fold)

    model_names = ['LinearRegression', 'Lasso', 'Ridge', 'DecisionTreeRegressor', 'KNeighborsRegressor',
                   'RandomForestRegressor', 'SVR', 'AdaBoostRegressor', 'GradientBoostingRegressor', 'XGBRegressor']
    scores = [score_lr, score_las, score_ri, score_dt, score_kn, score_rf, score_svr, score_ab, score_gb, score_xb]
    result = []
    for i in range(len(model_names)):
        score_mean = np.mean(scores[i])
        score_std = np.std(scores[i])
        m_names = model_names[i]
        temp = [m_names, score_mean, score_std]
        result.append(temp)
    kfold_df = pd.DataFrame(result, columns=['model_names', 'cv_score', 'cv_std'])
    return kfold_df.sort_values(by='cv_score', ascending=False)
print( )
print(kfoldCV(transformed_df.drop('charges in INR',axis = 1),transformed_df['charges in INR']))

def tuning(x,y,fold = 10):
   #parameters grids for different models
    param_las = {'alpha':[1e-15,1e-13,1e-11,1e-9,1e-7,1e-5,1e-4,1e-3,1e-1,0,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500]}
    param_rd = {'alpha':[1e-15,1e-13,1e-11,1e-9,1e-7,1e-5,1e-4,1e-3,1e-1,0,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500]}
    param_dtr = {'criterion':['squared_error','friedman_mse','absolute_error','poisson'],'max_depth':[3,5,7,9,11],'max_features':[1,2,3,4,5,6,7,'auto','log2', 'sqrt']}
    param_knn = {'weights':['uniform','distance'],'algorithm':['auto','ball_tree','kd_tree','brute']}
    param_svr = {'gamma':['scale','auto'],'C': [0.1,1,1.5,2]}
    param_rf = {'max_depth':[3,5,7,9,11],'max_features':[1,2,3,4,5,6,7,'auto','log2', 'sqrt'],'n_estimators':[50,100,150,200]}
    param_ab = {'n_estimators':[50,100,150,200],'learning_rate':[0.1,0.5,0.7,1,5,10,20,50,100]}
    param_gb = {'n_estimators':[50,100,150,200],'learning_rate':[0.1,0.5,0.7,1,5,10,20,50,100]}
    param_xb = {'eta':[0.1,0.5,10.7,1,5,10,20],'max_depth':[3,5,7,9,10],'gamma':[0,10,20,50],'reg_lambda':[0,1,3,5,7,10],'alpha':[0,1,3,5,7,10]}
    #Creating Model object
    tune_las = GridSearchCV(Lasso(),param_las,cv=fold)
    tune_rd = GridSearchCV(Ridge(),param_rd,cv=fold)
    tune_dtr = GridSearchCV(DecisionTreeRegressor(),param_dtr,cv=fold)
    tune_knn = GridSearchCV(KNeighborsRegressor(),param_knn,cv=fold)
    tune_svr = GridSearchCV(SVR(),param_svr,cv=fold)
    tune_rf = GridSearchCV(RandomForestRegressor(),param_rf,cv=fold)
    tune_ab = GridSearchCV(AdaBoostRegressor(),param_ab,cv=fold)
    tune_gb = GridSearchCV(GradientBoostingRegressor(),param_gb,cv=fold)
    tune_xb = GridSearchCV(XGBRegressor(),param_xb,cv=fold)
    #Model fitting
    tune_las.fit(x,y)
    tune_rd.fit(x,y)
    tune_dtr.fit(x,y)
    tune_knn.fit(x,y)
    tune_svr.fit(x,y)
    tune_rf.fit(x,y)
    tune_ab.fit(x,y)
    tune_gb.fit(x,y)
    tune_xb.fit(x,y)

    tune = [tune_rf,tune_xb,tune_gb,tune_las,tune_rd,tune_knn,tune_svr,tune_dtr,tune_ab]
    models = ['RF','XB','GB','lasso','RD','AB','KNN','SVR','DTR']
    for i in range(len(tune)):
        print('model:',models[i])
        print('Best_params:',tune[i].best_params_)


def cv_post_hpt(x,y,fold = 10):
    score_lr = cross_val_score(LinearRegression(),x,y,cv= fold)
    score_las = cross_val_score(Lasso(alpha= 30),x,y,cv=fold)
    score_rd = cross_val_score(Ridge(alpha=8),x,y,cv=fold)
    score_dt = cross_val_score(DecisionTreeRegressor( criterion= 'friedman_mse', max_depth = 5, max_features = 'auto'),x,y,cv= fold)
    score_kn = cross_val_score(KNeighborsRegressor(weights ='uniform' ,algorithm ='auto' ),x,y,cv=fold)
    score_rf = cross_val_score(RandomForestRegressor(max_depth= 9,max_features= 4 ,n_estimators= 200),x,y,cv=fold)
    score_svr = cross_val_score(SVR(gamma='scale' ,C= 2 ),x,y,cv=fold)
    score_ab = cross_val_score(AdaBoostRegressor(n_estimators= 50 ,learning_rate=0.1 ),x,y,cv=fold)
    score_gb = cross_val_score(GradientBoostingRegressor(n_estimators=100 ,learning_rate= 0.1),x,y,cv=fold)
    score_xb = cross_val_score(XGBRegressor(eta=0.1 ,max_depth= 5,gamma=0 ,reg_lambda = 3,alpha=0 ),x,y,cv=fold)

    model_names = ['LinearRegression','RandomForestRegressor','Lasso','Ridge','DecisionTreeRegressor','KNeighborsRegressor','SVR','AdaBoostRegressor','GradientBoostingRegressor','XGBRegressor']
    scores = [score_lr,score_rf,score_las,score_rd, score_dt,score_kn,score_svr,score_ab,score_gb,score_xb]
    result = []
    for i in range(len(model_names)):
        score_mean = np.mean(scores[i])
        score_std = np.std(scores[i])
        m_names = model_names[i]
        temp = [m_names,score_mean,score_std]
        result.append(temp)
    kfold_df = pd.DataFrame(result,columns=['model_names','cv_score','cv_std'])
    return kfold_df.sort_values(by='cv_score',ascending=False)


print( )
print(cv_post_hpt(transformed_df.drop('charges in INR',axis=1),transformed_df['charges in INR']))

labels = KMeans(n_clusters = 2,random_state=2)
clusters = labels.fit_predict(df.drop('charges in INR',axis=1))
s_plot = sns.scatterplot(x=df['sex'],y=df['charges in INR'],hue=clusters)

def clustering(x,tcol,clusters):
    column = list(set(list(x.columns)) - set(list('charges in INR')))
    #column = list(x.column)
    r = int(len(column)/2)
    if len(column)%2 == 0:
        r=r
    else:
        r += 1      #same as r+1
    f,ax = plt.subplots(r,2,figsize = (15,15))
    a = 0
    for row in range(r):
        for col in range(2):
            if a!= len(column):
                ax[row][col].scatter(x[tcol] , x[column[a]], c = clusters)
                ax[row][col].set_xlabel(tcol)
                ax[row][col].set_ylabel(column[a])
                a += 1
x = df.drop('charges in INR',axis = 1)
for col in x.columns:
    clustering(x , col , clusters)

plt.show()

new_df = df.join(pd.DataFrame(clusters,columns=['cluster']),how = 'left')
print(new_df.head().to_string())

new_f = new_df.groupby('cluster')['age'].agg(['mean','median'])
cluster_df = new_df.merge(new_f, on = 'cluster',how= 'left')
print(cluster_df.head().to_string())
print( )
print(multiple_models(cluster_df,'charges in INR'))
print( )
print(kfoldCV(cluster_df.drop('charges in INR',axis=1),cluster_df['charges in INR']))
print( )
print(cv_post_hpt(cluster_df.drop('charges in INR',axis=1),cluster_df['charges in INR']))
print( )
new__df = cluster_df
rfe = RFE(estimator = GradientBoostingRegressor())
rfe.fit(new__df.drop('charges in INR',axis=1),new__df['charges in INR'])
print(rfe.support_)
print( )
print(new__df.columns)
print( )
final_df = cluster_df[['age','bmi','children','smoker','charges in INR']]
print(cv_post_hpt(final_df.drop('charges in INR',axis=1),final_df['charges in INR']))


