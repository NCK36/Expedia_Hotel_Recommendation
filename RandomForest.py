# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import datetime
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

#Reading and loading the data
destinations = pd.read_csv("destinations.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv", nrows = 5000000)
train.head()

#Feature Engineering
#convert the date_time, srch_in and srch_co to datetime datatype
train["date_time"] = pd.to_datetime(train["date_time"])
train["year"] = train["date_time"].dt.year     #creating a year column
train["month"] = train["date_time"].dt.month   #creating a month column
train["srch_ci"] = pd.to_datetime(train["srch_ci"], infer_datetime_format = True, errors="coerce")
train["srch_co"] = pd.to_datetime(train["srch_co"], infer_datetime_format = True, errors="coerce")

#Doing same for test data
test["date_time"] = pd.to_datetime(test["date_time"])
test["year"] = test["date_time"].dt.year
test["month"] = test["date_time"].dt.month
test["srch_ci"] = pd.to_datetime(test["srch_ci"], infer_datetime_format = True, errors="coerce")
test["srch_co"] = pd.to_datetime(test["srch_co"], infer_datetime_format = True, errors="coerce")

#creating variable plan time and staying time in hotel
train["plan_time"] = ((train["srch_ci"]-train["date_time"])/np.timedelta64(1,"D")).astype(float)
train["hotel_stay"] = ((train["srch_co"]-train["srch_ci"])/np.timedelta64(1,"D")).astype(float)

test["plan_time"] = ((test["srch_ci"]-test["date_time"])/np.timedelta64(1,"D")).astype(float)
test["hotel_stay"] = ((test["srch_co"]-test["srch_ci"])/np.timedelta64(1,"D")).astype(float)

#fill missing values
train.isnull().sum()

#fill orig_destination_distance with mean of whole 
m = train.orig_destination_distance.mean()
train["orig_destination_distance"] = train.orig_destination_distance.fillna(m)
n = test.orig_destination_distance.mean()
test["orig_destination_distance"] = test.orig_destination_distance.fillna(n)

#droping string datatype
lst_drop = ["date_time","srch_ci","srch_co"]
train.drop(lst_drop, axis = 1, inplace = True)
test.drop(lst_drop, axis = 1, inplace = True)

#Using destination to generate features
#compress no of columns in destination to minimize run time
pca = PCA(n_components = 3)
dest_small = pca.fit_transform(destinations[["d{0}".format(i + 1) for i in range(149)]])
dest_small = pd.DataFrame(dest_small)
dest_small["srch_destination_id"] = destinations["srch_destination_id"]

train = train.join(dest_small, on="srch_destination_id", how='left', rsuffix="dest")
train = train.drop("srch_destination_iddest", axis=1)
test = test.join(dest_small, on="srch_destination_id", how='left', rsuffix="dest")
test = test.drop("srch_destination_iddest", axis=1)

#fill any missing values with -1
train.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)

#divide train into a new train and test subset
t1 = train[((train.year == 2014) & (train.month < 8))]
t2 = train[((train.year == 2014) & (train.month == 8))]

#assigning predictor and target
X_t1 = t1.drop(["hotel_cluster", "is_booking", "year", "cnt"], axis=1)
Y_t1 = t1["hotel_cluster"]
X_t2 = t2.drop(["hotel_cluster", "is_booking", "year", "cnt"], axis=1)
Y_t2 = t2["hotel_cluster"]

#using randomforest within train dataset to get cluster info
model_train = RandomForestClassifier(n_estimators = 31, max_depth = 10, random_state = 125)
model_train.fit(X_t1, Y_t1)

importance = model_train.feature_importances_
indices = np.argsort(importance)[::-1][:10]
importance[indices]


#storing different hotel clusters in a dictionary
len(model_train.classes_)
cluster_dict = {}
for (k,v) in enumerate(model_train.classes_):
    cluster_dict[k] = v

pred_train = model_train.predict_proba(X_t2)
a = pred_train.argsort(axis = 1)[:,-5:]       #gives the indices

#take the corresonding cluster of the 5 top indices
b = []
for i in a.flatten():
    b.append(cluster_dict.get(i))

cluster_pred_train = np.array(b).reshape(a.shape)

#divide the training set in predictor and target
target = train["hotel_cluster"]
train1 = train.drop(["hotel_cluster", "is_booking", "year", "cnt"], axis=1)
test1 = test.drop(["id", "year"], axis=1)

#split test1 in 5 subset
test_data = np.array_split(test1, 10)

#Using RandomForest
model = RandomForestClassifier(n_estimators = 31, max_depth = 10, random_state = 125)
model.fit(train1, target)

importance = model.feature_importances_
indices = np.argsort(importance)[::-1][:10]
importance[indices]

#predict on test dataset
pred = []
for i in range(len(test_data)):
    pred[i] = model.predict_proba(test_data[i])
prediction = lambda pred: [item for sublist in pred for item in sublist]

a = prediction.argsort(axis = 1)[:,-5:]

cluster_dict = {}
for (k,v) in enumerate(model_train.classes_):
    cluster_dict[k] = v

b = []
for i in a.flatten():
    b.append(cluster_dict.get(i))
cluster_pred = np.array(b).reshape(a.shape)

cluster_pred = list(map(lambda x: " ".join(map(str,x)), cluster_pred))

id = pd.Series(test["id"])

out = np.column_stack((id, cluster_pred))
output = pd.DataFrame(out)
output.to_csv("out_data.csv", index=False)
