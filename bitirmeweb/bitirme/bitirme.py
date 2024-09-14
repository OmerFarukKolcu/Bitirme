import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

veriler = pd.read_csv('tagss.csv')

tags=veriler.iloc[:,0:1].values
le=preprocessing.LabelEncoder()
tags[:,0]=le.fit_transform(veriler.iloc[:,0])
veriler=veriler.assign(label_encoder=tags)
yedek=veriler
yedek1=veriler.head()
likes_likes=yedek.iloc[:,2:3]
tagsslabel=yedek.iloc[:,4:]
tagssss=yedek.iloc[:,0:1]
likesdeneme=likes_likes.iloc[0:20]
tagsdeneme=tagsslabel[0:20]

"""----------------------------Basit Doğrusal Regresyon-----------
x_train,x_test,y_train,y_test=train_test_split(tagsslabel,likes_likes,test_size=0.33,random_state=0,shuffle=True)
lr1=LinearRegression()
lr1.fit(x_train,y_train)
tahmin=lr1.predict(x_test)
print(tahmin)
print(y_test)
Tahmin sonuçları
[[38591.17259367]
 [30687.97018942]
 [38052.92632546]
 ...
 [30098.999142  ]
 [39702.89067122]
 [34226.02353887]]
y_test sonuçları
       likes
18270  66270
7104    1923
6956    8542
841     3307
11841  31301
     ...
17576   2638
18292  12287
7286     527
"""
plt.show()

#decision tree
#x=veriler.iloc[:,1:2]#eğitim sev
x=likesdeneme
#y=veriler.iloc[:,2:]#maas
y=tagsdeneme
X=x.values
Y=y.values
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
plt.scatter(X, Y,color='red')
plt.plot(Y,r_dt.predict(X),color='blue')
print(int(r_dt.predict([[127794]])))





