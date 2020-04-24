# import statements for packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# import warnings to ignore deprecation warning
import warnings
warnings.filterwarnings('ignore')

# load cvs files of data using pandas
df_train = pd.read_csv('./data/train.csv', index_col=False)
df_test = pd.read_csv('./data/test.csv', index_col=False)

# Training Data Presentation

# variables and info
print(df_train.info())

# for SalePrice
# descriptive stats summary
print(df_train['SalePrice'].describe())

# histogram
sns.distplot(df_train['SalePrice'])
plt.show()

# positively skewed
print("Skewness: %f" % df_train['SalePrice'].skew())

# scatter plots of SalePrice against numerical variables
for var in ['GrLivArea', 'TotalBsmtSF', 'LotArea']:
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
    plt.show()

# outliers according to scatter plots
df_train.drop(df_train[(df_train['GrLivArea'] > 4000) & (df_train['SalePrice'] < 300000)].index, inplace=True)
# df_train.drop(df_train[(df_train['LotArea'] > 100000)].index, inplace=True)
# df_train.drop(df_train[(df_train['TotalBsmtSF'] > 6000)].index, inplace=True)

# boxplots of SalePrice against some categorical variables
for var in ['OverallQual', 'YearBuilt', 'TotRmsAbvGrd']:
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    f, ax = plt.subplots()
    fig = sns.boxplot(x=var, y='SalePrice', data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.show()

# correlation matrix heatmap
corr_matrix = df_train.corr()
f, ax = plt.subplots()
sns.heatmap(corr_matrix, square=True);
plt.show()

# correlation matrix of top 10 variables most correlated to SalePrice
k = 10
cols = corr_matrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                 xticklabels=cols.values)
plt.show()

# Data Cleaning

# missing values overview
missing_total = df_train.isnull().sum().sort_values(ascending=False)
missing_percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing = pd.concat([missing_total, missing_percent], axis=1, keys=['Count', 'Percent'])
print(missing[missing_total > 0])

# join train and test dataset for processing
train_price = df_train['SalePrice']  # save SalePrice for future; drop it now to make columns the same
df_train.drop('SalePrice', axis=1, inplace=True)
train_num = len(df_train)  # length of train dataset
both = pd.concat([df_train, df_test]).reset_index(drop=True)
both.drop(['Id'], axis=1, inplace=True)

# drop features which have too many missing values
for var in ['Alley', 'PoolQC', 'Fence', 'MiscFeature']:
    both.drop(var, axis=1, inplace=True)

# fill in missing values based on nature of the variable
# numerical
for var in ['MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'BsmtFinSF2', 'BsmtFinSF1', 'GarageArea']:
    both[var] = both[var].fillna(0)
# categorical
for var in ['FireplaceQu', 'GarageQual', 'GarageCond', 'GarageFinish', 'GarageYrBlt', 'GarageType', 'BsmtExposure','BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType']:
    both[var] = both[var].fillna('None')
# mode
for var in ['MSZoning', 'BsmtFullBath', 'BsmtHalfBath', 'Utilities', 'Functional', 'Electrical', 'KitchenQual', 'SaleType', 'Exterior1st', 'Exterior2nd']:
    both[var] = both[var].fillna(both[var]).mode()[0]
# LotFrontage
both['LotFrontage'] = both.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# no missing value left at this point

# features processing
log_price = np.log(train_price) # log transformation on SalePrice

# Label Encoding
le = LabelEncoder()
for var in ['OverallQual','OverallCond','YearBuilt','YearRemodAdd', 'ExterQual','ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure','BsmtFinType1', 'BsmtFinType2','HeatingQC',
            'CentralAir','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd' ,'Fireplaces','FireplaceQu','GarageYrBlt',
            'GarageFinish','GarageCars','MiscVal','MoSold','YrSold', 'GarageQual', 'GarageCond']:
    both[var] = le.fit_transform(both[var].astype(str))

# One-hot Encoding (get_dummies)
for var in ['MSSubClass', 'MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle',
            'RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','Electrical','Functional','GarageType','PavedDrive','SaleType','SaleCondition']:
    both[var] = pd.get_dummies(both[var])

# log transformation (normalization)
for var in ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea','WoodDeckSF',
            'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea']:
    both.loc[both[var]>0, var] = np.log(both[var]) # cannot log transform 0

# retrieve train dataset
processed_train = both[:train_num]
processed_test = both[train_num:]

# Model Prediction

# finding optimal parameters for regression algorithms using k-fold cross validation
# k = 10

# XGBoost
# score = []
# alpha = [1,2,3,4,5,6]
# for a in alpha:
#     clf = XGBRegressor(max_depth=a)
#     result = np.sqrt(-cross_val_score(clf, processed_train, log_price, cv=k, scoring='neg_mean_squared_error'))
#     score.append(np.mean(result))
# print(score) # max_depth = 5 is optimal for XGBoost

# Random Forest
# score = []
# alpha = [0.1,0.3,0.5,0.7,0.99]
# for a in alpha:
#     clf = RandomForestRegressor(n_estimators=200, max_features=a)
#     result = np.sqrt(-cross_val_score(clf, processed_train, log_price, cv=k, scoring='neg_mean_squared_error'))
#     score.append(np.mean(result))
# print(score) # max_feature = 0.5 is optimal for Random Forest

# Ridge
# score = []
# alpha = np.logspace(-3, 2, 100) # choose 100 numbers from 0.001 to 100
# for a in alpha:
#     clf = Ridge(a)
#     result = np.sqrt(-cross_val_score(clf, processed_train, log_price, cv=k, scoring='neg_mean_squared_error'))
#     score.append(np.mean(result))
# print(alpha[np.argmin(score)]) # optimal when a is about 1.7

# prediction and RMSE score

# XGBoost
XGBoost = XGBRegressor(max_depth=5)
XGBoost.fit(processed_train, log_price)
XGBoost_price = XGBoost.predict(processed_train)
XGBoost_score = np.sqrt(mean_squared_error(XGBoost_price, log_price))
print('RMSE of XGBoost：{}'.format(XGBoost_score))

# Random Forest
RF = RandomForestRegressor(n_estimators=200, max_features=0.5)
RF.fit(processed_train, log_price)
RF_price = RF.predict(processed_train)
RF_score = np.sqrt(mean_squared_error(RF_price, log_price))
print('RMSE of Random Forest：{}'.format(RF_score))

# Ridge
Ridge = Ridge(1.7)
Ridge.fit(processed_train, log_price)
Ridge_price = Ridge.predict(processed_train)
Ridge_score = np.sqrt(mean_squared_error(Ridge_price, log_price))
print('RMSE of Ridge：{}'.format(Ridge_score))

# Therefore, by comparison, Random Forest produces the least RMSE and thus is the best regressor among 3

# Bonus: submission to Kaggle using RF
RF_sub = RF
RF_price_sub = RF_sub.predict(processed_test)
submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': np.expm1(RF_price_sub)})
submission.to_csv('./data/submission.csv', index=False)
# score from Kaggle is 0.14329

