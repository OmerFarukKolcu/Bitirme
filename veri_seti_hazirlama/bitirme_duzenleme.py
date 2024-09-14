import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

veriler = pd.read_csv('USvideos_modified.csv')


print(len(veriler['subscriber']))
video_icerigi=veriler['tags']
kanal_icerigi=veriler['title']
veri=veriler['subscriber'].mean()
veri=int(veri)
veriler['subscriber']=veriler['subscriber'].fillna(veri)
print(veriler.isna().sum())
yedek=veriler
"""egitim = pd.read_csv('egitim_icerik.csv')


izlenme_say = egitim.iloc[:, 6:7]
#izlenme_say=veriler['views']
print("izlenme say",izlenme_say)
#izlenme_say=yedek.drop('views',1)
abone_say = egitim.iloc[:, -1]

#abone_say=veriler['subscriber']
abone_say=abone_say.astype('int64')
print("abone say",abone_say)
print("abone sayısı",type(abone_say))
print("izlenme say",type(izlenme_say))
print("mın deger=",izlenme_say.min())


#abone_say=yedek.drop('subscriber',1)
        #print("sub count",sub_count)
        #print("izlneme",izlneme)
#a=np.where(abone_say.isna().any())[0]
        #print("izlenme eksik veri indexi",a)
        #print("izlenme eksik veri",izlneme.isna().sum())
#b = np.where(izlenme_say.isna().any())[0]
        #print("abone eksik veri indexi", a)
        #print("abone eksik veri", sub_count.isna().sum())
x_train, x_test, y_train, y_test = train_test_split(abone_say, izlenme_say, test_size=0.33,
                                                            random_state=0)
x_test=np.asarray(x_test)
y_test=np.asarray(y_test)
x_train=np.asarray(x_train)
y_train=np.asarray(y_train)

x_test=x_test.reshape(-1,1)
y_test=y_test.reshape(-1,1)
x_train=x_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
tahmin=lr.predict(x_test)
print(tahmin)
print(lr.predict([[3947725]]))"""

music_icerik = pd.DataFrame(
    columns=["video_id", "last_trending_date", "publish_date", "publish_hour", "category_id", "channel_title",
             "views", "likes", "dislikes", "comment_count", "comments_disabled", "ratings_disabled",
             "tag_appeared_in_title_count", "tag_appeared_in_title", "title", "tags", "description",
             "trend_day_count", "trend.publish.diff", "trend_tag_highest", "trend_tag_total", "tags_count",
             "subscriber"])
yemek_icerik = pd.DataFrame(
    columns=["video_id", "last_trending_date", "publish_date", "publish_hour", "category_id", "channel_title",
             "views", "likes", "dislikes", "comment_count", "comments_disabled", "ratings_disabled",
             "tag_appeared_in_title_count", "tag_appeared_in_title", "title", "tags", "description",
             "trend_day_count", "trend.publish.diff", "trend_tag_highest", "trend_tag_total", "tags_count",
             "subscriber"])
siyasi_icerik = pd.DataFrame(
    columns=["video_id", "last_trending_date", "publish_date", "publish_hour", "category_id", "channel_title",
             "views", "likes", "dislikes", "comment_count", "comments_disabled", "ratings_disabled",
             "tag_appeared_in_title_count", "tag_appeared_in_title", "title", "tags", "description",
             "trend_day_count", "trend.publish.diff", "trend_tag_highest", "trend_tag_total", "tags_count",
             "subscriber"])
egitim_icerik = pd.DataFrame(
    columns=["video_id", "last_trending_date", "publish_date", "publish_hour", "category_id", "channel_title",
             "views", "likes", "dislikes", "comment_count", "comments_disabled", "ratings_disabled",
             "tag_appeared_in_title_count", "tag_appeared_in_title", "title", "tags", "description",
             "trend_day_count", "trend.publish.diff", "trend_tag_highest", "trend_tag_total", "tags_count",
             "subscriber"])
spor_icerik = pd.DataFrame(
    columns=["video_id", "last_trending_date", "publish_date", "publish_hour", "category_id", "channel_title",
             "views", "likes", "dislikes", "comment_count", "comments_disabled", "ratings_disabled",
             "tag_appeared_in_title_count", "tag_appeared_in_title", "title", "tags", "description",
             "trend_day_count", "trend.publish.diff", "trend_tag_highest", "trend_tag_total", "tags_count",
             "subscriber"])
eglence_icerik = pd.DataFrame(
    columns=["video_id", "last_trending_date", "publish_date", "publish_hour", "category_id", "channel_title",
             "views", "likes", "dislikes", "comment_count", "comments_disabled", "ratings_disabled",
             "tag_appeared_in_title_count", "tag_appeared_in_title", "title", "tags", "description",
             "trend_day_count", "trend.publish.diff", "trend_tag_highest", "trend_tag_total", "tags_count",
             "subscriber"])
oyun_icerik = pd.DataFrame(
    columns=["video_id", "last_trending_date", "publish_date", "publish_hour", "category_id", "channel_title",
             "views", "likes", "dislikes", "comment_count", "comments_disabled", "ratings_disabled",
             "tag_appeared_in_title_count", "tag_appeared_in_title", "title", "tags", "description",
             "trend_day_count", "trend.publish.diff", "trend_tag_highest", "trend_tag_total", "tags_count",
             "subscriber"])

for i in range(len(video_icerigi)):
    icerik = video_icerigi[i]
    if type(video_icerigi[i]) !=str:
        
        # nan verilerin yerine kanal açıklamaları yazıldı
        video_icerigi[i] = kanal_icerigi[i]
        icerik = video_icerigi[i]   
        print("eksik veri bulundu:", video_icerigi[i])
    if "music" in icerik or "pop" in icerik or "rap" in icerik or "musical" in icerik or "classic music" in icerik:
        music = veriler.iloc[[i]]
        music_icerik = pd.concat([music_icerik, music])
    if "food" in icerik or "kitchen" in icerik or "cook" in icerik or "cooking" in icerik  :
        yemek = veriler.iloc[[i]]
        yemek_icerik = pd.concat([yemek_icerik, yemek])
    if "political" in icerik or "democrat" in icerik or "elections" in icerik or "war" in icerik or "prime minister" in icerik or "president" in icerik:
        siyasi = veriler.iloc[[i]]
        siyasi_icerik = pd.concat([siyasi_icerik, siyasi])
    if "education" in icerik or "lesson" in icerik or "math" in icerik or "Math" in icerik or "exam" in icerik or "language" in icerik:
        egitim = veriler.iloc[[i]]
        egitim_icerik = pd.concat([egitim, egitim_icerik])

    if "score" in icerik or "spor" in icerik or "sport" in icerik or "football" in icerik or "basketball" in icerik or "Olympic" in icerik or "NFL" in icerik or "nba" in icerik or "NBA" in icerik:
        spor = veriler.iloc[[i]]
        spor_icerik = pd.concat([spor_icerik, spor])

    if "funny" in icerik or "film" in icerik or "trailer" in icerik or "stand up" in icerik or "standup" in icerik or "comedy" in icerik:
        eglence = veriler.iloc[[i]]
        eglence_icerik = pd.concat([eglence_icerik, eglence])
    if "videogame" in icerik or "video game" in icerik or "csgo" in icerik or "aim" in icerik or "LOL" in icerik or "riot" in icerik or "gaming setup" in icerik:
        oyun = veriler.iloc[[i]]
        oyun_icerik = pd.concat([oyun, oyun_icerik])
print(type(video_icerigi[0]))
music_icerik.to_csv('music_icerik.csv',mode='w',index=False,header=True)
yemek_icerik.to_csv('yemek_icerik.csv',mode='w',index=False,header=True)
siyasi_icerik.to_csv('siyasi_icerik.csv',mode='w',index=False,header=True)
egitim_icerik.to_csv('egitim_icerik.csv',mode='w',index=False,header=True)
spor_icerik.to_csv('spor_icerik.csv',mode='w',index=False,header=True)
eglence_icerik.to_csv('eglence_icerik.csv',mode='w',index=False,header=True)
oyun_icerik.to_csv('oyun_icerik.csv',mode='w',index=False,header=True)

#for i in range(len(video_icerigi)):













