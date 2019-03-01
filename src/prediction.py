import networkx as nx
import operator
import pandas as pd
import numpy as np
import os
from math import isnan
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import re
import matplotlib.pyplot as plt


def merge_to_df(func, df, G):
  features = func(G)
  print(features)
  features = {k: features[k] for k in features if not isnan(float(k))}
  
  features_df = pd.DataFrame.from_dict(features, orient="index",columns=[str(func.__name__)]).reset_index().rename(columns={"index":"Id"})
  df = pd.merge(left=df, right=features_df, on="Id", how='left')
  return df

if __name__ == "__main__":
  print(__file__)
  G = nx.read_gpickle(os.getcwd()+"/data/Users_comments2.pkl")
  print(len(G.nodes))

  try:
    df1 = pd.read_pickle(os.getcwd()+"/data/df_with_betweenness.pkl")
    print(df1.shape)
    df = pd.read_pickle(os.getcwd()+"/data/df_with_betweenness2.pkl")
    print(df.shape)
    df = pd.concat([df1, df])
  except e:
    print(e)
    df = pd.read_pickle(os.getcwd()+"/data/Usersweek_2.pkl")
    df = merge_to_df(nx.degree_centrality, df, G)
    pd.to_pickle(df, "df_with_degree2.pkl")
    df = merge_to_df(nx.closeness_centrality, df, G)
    pd.to_pickle(df, "df_with_closeness2.pkl")
    df = merge_to_df(nx.betweenness_centrality, df, G)
    pd.to_pickle(df, "df_with_betweenness2.pkl")

    
    # cleaning up my shit
  df = df.rename(columns=lambda x: re.sub('^.*(?=\s(?=(\S*_\S)))','',x))
  df = df.rename(columns=lambda x: re.sub('\S.\s.*','',x))
  df.columns = df.columns.str.strip()
  # cleaned!

  df = pd.merge(left=df, right=pd.DataFrame.from_dict( nx.degree(G),orient="columns").rename(columns={0: "Id", 1:"Degree"}), on="Id", how="left")


  df = df.drop(["AboutMe", "DisplayName", "AccountId", "LastAccessDate", "ProfileImageUrl", "WebsiteUrl", "CreationDate", "Id"], axis="columns")
  le = preprocessing.LabelEncoder()
  df["Location"] = le.fit_transform(df["Location"].fillna('helvetti'))
  df = df.groupby(level=0, axis=1).sum()
  df = df.dropna()

  y = df["Reputation"]
  X = df.drop(["Reputation"], axis="columns")

  print(max(y), min(y))
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

  regr = RandomForestRegressor(max_depth=25, n_estimators=250)
  regr.fit(X_train, y_train)
  print(X.columns)
  print(regr.feature_importances_)
  prediction = regr.predict(X_test)
  print("Random forest")
  print("(Predictions, True): ")
  print(list(zip(prediction[25:50], np.array(y_test)[25:50])))
  print("MSE: ")
  print(mean_squared_error(prediction, y_test))
  prediction, y_test = [list(t) for t in zip(*sorted(zip(prediction, np.array(y_test))))]
  
  plt.figure()
  plt.plot(np.array(range(len(prediction))), prediction,'r', np.array(range(len(prediction))), np.array(y_test), 'b')
  plt.legend(["True", "Prediction"])
  plt.figure()
  plt.plot(np.array(range(len(prediction))), abs(np.subtract(y_test, prediction)))
  correct = 0
  spotted = 0
  false = 0
  superstars = 0
  for p, y in zip(prediction, y_test):
    if y > 2000:
      superstars += 1
    if y > 2000 and p > 2000:
      spotted += 1
      correct += 1
    if y <= 2000 and p <= 2000:
      correct += 1
    else:
      false += 1
  print(correct, false)
  print(superstars)
  print(spotted)
  precision = correct/ (correct + false)
  recall = spotted/superstars
  print("precision: ", precision)
  print("recall: ", recall)
  print("F1: ", 2 * (precision * recall) / (precision + recall)
)
  # regr = LinearRegression().fit(X_train, y_train)
  # prediction = regr.predict(X_test)
  # print("Linear Regression")
  # print("Predictions: ")
  # print(prediction)
  # print("True labels: ")
  # print(np.array(y_test))
  # print("MSE: ")
  # print(mean_squared_error(prediction, y_test))