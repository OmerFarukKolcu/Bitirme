from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Create your views here.
from django.http import HttpResponse

def index(request):
    return render(request,'pages/son.html')
def about(request):
    if request.method == 'POST':
        kullanici_sub_say=request.POST["user"]
        #print(type(kullanici_sub_say))
        #kullanıcı sayısı girilmez ise:
        try:
            y = int(kullanici_sub_say)
        except ValueError:
            return render(request, 'pages/son.html')

        kullanici_sub_say=int(kullanici_sub_say)
        ilgi_alani=request.POST["alan"]
        print("subsay=", type(kullanici_sub_say))
        print("ilgi alanı=",ilgi_alani)
        kullanici_sub_say=int(kullanici_sub_say)

    veriler = pd.read_csv('pages/static/USvideos_modified.csv')
    veri = veriler['subscriber'].mean()
    print("ortalama=", veri)
    veri = int(veri)
    veriler['subscriber'] = veriler['subscriber'].fillna(veri)
    print(veriler.isna().sum())
    kanal_icerigi = veriler['channel_title']
    video_icerigi = veriler['tags']
    eglence = pd.read_csv('pages/static/eglence_icerik.csv')
    egitim = pd.read_csv('pages/static/egitim_icerik.csv')
    music = pd.read_csv('pages/static/music_icerik.csv')
    oyun = pd.read_csv('pages/static/oyun_icerik.csv')
    siyasi = pd.read_csv('pages/static/siyasi_icerik.csv')
    spor = pd.read_csv('pages/static/spor_icerik.csv')
    yemek = pd.read_csv('pages/static/yemek_icerik.csv')
    print("Boyutlar")
    print("eglence",len(eglence['tags']))
    print("egitim",len(egitim['tags']))
    print("music",len(music['tags']))
    print("oyun",len(oyun['tags']))
    print("siyasi",len(siyasi['tags']))
    print("spor",len(spor['tags']))
    print("yemek",len(yemek['tags']))
    kontrol=0
    #eksik veriler ortalama ile tamamlandı---------------------
    eglence_ortalama=eglence["subscriber"].mean()
    egitim_ortalama = egitim["subscriber"].mean()
    music_ortalama = music["subscriber"].mean()
    oyun_ortalama = oyun["subscriber"].mean()
    siyasi_ortalama = siyasi["subscriber"].mean()
    spor_ortalama = spor["subscriber"].mean()
    yemek_ortalama = yemek["subscriber"].mean()
    #print("egitim abone ortalaması=",egitim_ortalama)
    eglence['subscriber']=eglence["subscriber"].fillna(eglence_ortalama)
    egitim['subscriber'] = egitim["subscriber"].fillna(egitim_ortalama)
    music['subscriber'] = music["subscriber"].fillna(music_ortalama)
    oyun['subscriber'] = oyun["subscriber"].fillna(oyun_ortalama)
    siyasi['subscriber'] = siyasi["subscriber"].fillna(siyasi_ortalama)
    spor['subscriber'] = spor["subscriber"].fillna(spor_ortalama)
    yemek['subscriber'] = yemek["subscriber"].fillna(yemek_ortalama)
    #---------------------------





    if ilgi_alani=="eglence":
        kontrol = 1
        # makine öğrenmesi:
        izlenme_say = eglence.iloc[:, 6:7]
        abone_say = eglence.iloc[:, -1]
        abone_say = abone_say.astype('int64')
        # print("sub count",sub_count)
        # print("izlneme",izlneme)
        # print("izlenme eksik veri indexi",a)
        # print("izlenme eksik veri",izlneme.isna().sum())
        # print("abone eksik veri indexi", a)
        # print("abone eksik veri", sub_count.isna().sum())
        x_train, x_test, y_train, y_test = train_test_split(abone_say, izlenme_say, test_size=0.33,
                                                            random_state=0, shuffle=True)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        x_test = x_test.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        tahmin = lr.predict(x_test)

        # ------------------------decision tree
        from sklearn.tree import DecisionTreeRegressor
        X = izlenme_say.values
        Y = abone_say.values
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        r_dt = DecisionTreeRegressor(random_state=0)
        r_dt.fit(Y, X)
        abone_alt_sinir = abone_say.min()
        abone_ust_sinir = abone_say.max()
        izlenme_alt_sinir = izlenme_say.min()
        izlenme_alt_sinir = izlenme_alt_sinir.item()
        izlenme_ust_sinir = izlenme_say.max()
        izlenme_ust_sinir = izlenme_ust_sinir.item()
        print("izlenme:\nust sınır", izlenme_ust_sinir)
        print("alt sınır", izlenme_alt_sinir)
        print("abone:\nust sınır", abone_ust_sinir)
        print("alt sınır", abone_alt_sinir.item())
        if kullanici_sub_say < abone_ust_sinir:
            predict_egitim = r_dt.predict([[kullanici_sub_say]])
            predict_egitim = int(predict_egitim)
        else:
            predict_egitim = lr.predict([[kullanici_sub_say]])
            predict_egitim = int(predict_egitim)
            if predict_egitim < 0:
                predict_egitim = predict_egitim * (-1)
        print("tahmin değeri:", predict_egitim)

        if predict_egitim > izlenme_alt_sinir and predict_egitim < izlenme_ust_sinir:
            fikir_alt_sinir = predict_egitim - 5000
            fikir_ust_sinir = predict_egitim + 5000
            x_ideal = eglence[eglence['views'].between(fikir_alt_sinir, fikir_ust_sinir)]
            print("x ideal=", x_ideal['title'])
            oneri_title=[]
            oneri_views=[]
            oneri_likes=[]
            oneri_dislikes=[]
            oneri_paylasim_zamani=[]
            oneri_trend_day=[]
            if len(x_ideal['title']) >= 3:
                for i in range(3):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    print("öneri views",type(oneri_views))
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                dict = {
                    'tahmim': predict_egitim,
                    'views0': oneri_views[0].item(),
                    'views1': oneri_views[1].item(),
                    'likes0': oneri_likes[0].item(),
                    'likes1': oneri_likes[1].item(),
                    'dislike0': oneri_dislikes[0].item(),
                    'dislike1': oneri_dislikes[1].item(),
                    'tags1': oneri_title[0].item(),
                    'tags2': oneri_title[1].item(),
                    'tags3': oneri_title[2].item(),
                    'views2': oneri_views[2].item(),
                    'likes2': oneri_likes[2].item(),
                    'dislike2': oneri_dislikes[2].item(),
                    'paylasim0': oneri_paylasim_zamani[0].item(),
                    'paylasim1': oneri_paylasim_zamani[1].item(),
                    'paylasim2': oneri_paylasim_zamani[2].item(),
                    'trend_day0': oneri_trend_day[0].item(),
                    'trend_day1': oneri_trend_day[1].item(),
                    'trend_day2': oneri_trend_day[2].item(),

                }
                return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 2:
                  for i in range(2):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                  dict = {
                        'tahmim': predict_egitim,
                        'views0': oneri_views[0].item(),
                        'views1': oneri_views[1].item(),
                        'likes0': oneri_likes[0].item(),
                        'likes1': oneri_likes[1].item(),
                        'dislike0': oneri_dislikes[0].item(),
                        'dislike1': oneri_dislikes[1].item(),
                        'tags1': oneri_title[0].item(),
                        'tags2': oneri_title[1].item(),
                        'paylasim0': oneri_paylasim_zamani[0].item(),
                        'paylasim1': oneri_paylasim_zamani[1].item(),
                        'tags3': "",
                        'views2': "",
                        'likes2': "",
                        'dislike2': "",
                        'paylasim2': "",
                        'trend_day0': oneri_trend_day[0].item(),
                        'trend_day1': oneri_trend_day[1].item(),
                        'trend_day2': "",

                  }
                  return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 1:
                  for i in range(1):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                  dict = {
                        'tahmim': predict_egitim,
                        'views0': oneri_views[0].item(),
                        'views1': "",
                        'likes0': oneri_likes[0].item(),
                        'likes1': "",
                        'dislike0': oneri_dislikes[0].item(),
                        'dislike1': "",
                        'tags1': oneri_title[0].item(),
                        'tags2': "",
                        'tags3': "",
                        'views2': "",
                        'likes2': "",
                        'dislike2': "",
                        'paylasim0': oneri_paylasim_zamani[0].item(),
                        'paylasim1': "",
                        'paylasim2': "",
                        'trend_day0': oneri_trend_day[0].item(),
                        'trend_day1': "",
                        'trend_day2': "",

                    }
                  return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 0:
                for i in range(3):
                    """oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])"""
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                    dict = {
                        'tahmim': predict_egitim,
                        'views0': "",
                        'views1': "",
                        'likes0': "",
                        'likes1': "",
                        'dislike0': "",
                        'dislike1': "",
                        'tags1': "No suggestions found",
                        'tags2': "No suggestions found",
                        'tags3': "No suggestions found",
                        'views2': "",
                        'likes2': "",
                        'dislike2': "",
                        'paylasim0': "",
                        'paylasim1': "",
                        'paylasim2': "",
                        'trend_day0':"",
                        'trend_day1': "",
                        'trend_day2': "",

                    }
                    return render(request, 'pages/about.html', dict)


            """oneri_title0 = x_ideal['title'].iloc[[0]]
            oneri_title1 = x_ideal['title'].iloc[[1]]
            oneri_views0 = x_ideal['views'].iloc[[0]]
            oneri_views1 = x_ideal['views'].iloc[[1]]
            oneri_likes0 = x_ideal['likes'].iloc[[0]]
            oneri_likes1 = x_ideal['likes'].iloc[[1]]
            oneri_dislikes0 = x_ideal['dislikes'].iloc[[0]]
            oneri_dislikes1 = x_ideal['dislikes'].iloc[[1]]"""









    if ilgi_alani=="egitim":

        kontrol = 1
        #makine öğrenmesi:
        izlenme_say = egitim.iloc[:, 6:7]
        abone_say = egitim.iloc[:, -1]
        abone_say = abone_say.astype('int64')
        #print("sub count",sub_count)
        #print("izlneme",izlneme)
        #print("izlenme eksik veri indexi",a)
        #print("izlenme eksik veri",izlneme.isna().sum())
        #print("abone eksik veri indexi", a)
        #print("abone eksik veri", sub_count.isna().sum())
        x_train, x_test, y_train, y_test = train_test_split(abone_say, izlenme_say, test_size=0.33,
                                                            random_state=0, shuffle=True)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        x_test = x_test.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        tahmin = lr.predict(x_test)

        # ------------------------decision tree
        from sklearn.tree import DecisionTreeRegressor
        X = izlenme_say.values
        Y = abone_say.values
        X=X.reshape(-1, 1)
        Y=Y.reshape(-1, 1)
        r_dt = DecisionTreeRegressor(random_state=0)
        r_dt.fit(Y, X)
        abone_alt_sinir=abone_say.min()
        abone_ust_sinir = abone_say.max()
        izlenme_alt_sinir=izlenme_say.min()
        izlenme_alt_sinir=izlenme_alt_sinir.item()
        izlenme_ust_sinir=izlenme_say.max()
        izlenme_ust_sinir=izlenme_ust_sinir.item()
        print("izlenme:\nust sınır",izlenme_ust_sinir)
        print("alt sınır",izlenme_alt_sinir)
        print("abone:\nust sınır", abone_ust_sinir)
        print("alt sınır", abone_alt_sinir.item())
        if kullanici_sub_say<abone_ust_sinir:
            predict_egitim = r_dt.predict([[kullanici_sub_say]])
            predict_egitim=int(predict_egitim)
        else:
            predict_egitim=lr.predict([[kullanici_sub_say]])
            predict_egitim = int(predict_egitim)
            if predict_egitim<0:
                predict_egitim=predict_egitim*(-1)
        print("tahmin değeri:",predict_egitim)

        if predict_egitim>izlenme_alt_sinir and predict_egitim<izlenme_ust_sinir:
            fikir_alt_sinir=predict_egitim-10000
            fikir_ust_sinir = predict_egitim + 10000
            x_ideal = egitim[egitim['views'].between(fikir_alt_sinir, fikir_ust_sinir)]
            print("x ideal=",x_ideal['title'])
            """oneri_title0=x_ideal['title'].iloc[[0]]
            oneri_title1 = x_ideal['title'].iloc[[1]]
            oneri_views0=x_ideal['views'].iloc[[0]]
            oneri_views1 = x_ideal['views'].iloc[[1]]
            oneri_likes0=x_ideal['likes'].iloc[[0]]
            oneri_likes1= x_ideal['likes'].iloc[[1]]
            oneri_dislikes0 = x_ideal['dislikes'].iloc[[0]]
            oneri_dislikes1 = x_ideal['dislikes'].iloc[[1]]"""
            oneri_title = []
            oneri_views = []
            oneri_likes = []
            oneri_dislikes = []
            oneri_paylasim_zamani = []
            oneri_trend_day=[]
            if len(x_ideal['title']) >= 3:
                print("x ideal için 3 buldu")
                for i in range(3):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    print(oneri_trend_day)

                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                dict = {
                    'tahmim': predict_egitim,
                    'views0': oneri_views[0].item(),
                    'views1': oneri_views[1].item(),
                    'likes0': oneri_likes[0].item(),
                    'likes1': oneri_likes[1].item(),
                    'dislike0': oneri_dislikes[0].item(),
                    'dislike1': oneri_dislikes[1].item(),
                    'tags1': oneri_title[0].item(),
                    'tags2': oneri_title[1].item(),
                    'tags3': oneri_title[2].item(),
                    'views2': oneri_views[2].item(),
                    'likes2': oneri_likes[2].item(),
                    'dislike2': oneri_dislikes[2].item(),
                    'paylasim0': oneri_paylasim_zamani[0].item(),
                    'paylasim1': oneri_paylasim_zamani[1].item(),
                    'paylasim2': oneri_paylasim_zamani[2].item(),
                    'trend_day0': oneri_trend_day[0].item(),
                    'trend_day1': oneri_trend_day[1].item(),
                    'trend_day2': oneri_trend_day[2].item(),

                }
                return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 2:
                print("x ideal için 2 buldu")
                for i in range(2):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                dict = {
                    'tahmim': predict_egitim,
                    'views0': oneri_views[0].item(),
                    'views1': oneri_views[1].item(),
                    'likes0': oneri_likes[0].item(),
                    'likes1': oneri_likes[1].item(),
                    'dislike0': oneri_dislikes[0].item(),
                    'dislike1': oneri_dislikes[1].item(),
                    'tags1': oneri_title[0].item(),
                    'tags2': oneri_title[1].item(),
                    'paylasim0': oneri_paylasim_zamani[0].item(),
                    'paylasim1': oneri_paylasim_zamani[1].item(),
                    'tags3': "",
                    'views2': "",
                    'likes2': "",
                    'dislike2': "",
                    'paylasim2': "",
                    'trend_day0': oneri_trend_day[0].item(),
                    'trend_day1': oneri_trend_day[1].item(),
                    'trend_day2': "",

                }
                return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 1:
                for i in range(1):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                dict = {
                    'tahmim': predict_egitim,
                    'views0': oneri_views[0].item(),
                    'views1': "",
                    'likes0': oneri_likes[0].item(),
                    'likes1': "",
                    'dislike0': oneri_dislikes[0].item(),
                    'dislike1': "",
                    'tags1': oneri_title[0].item(),
                    'tags2': "",
                    'tags3': "",
                    'views2': "",
                    'likes2': "",
                    'dislike2': "",
                    'paylasim0': oneri_paylasim_zamani[0].item(),
                    'paylasim1': "",
                    'paylasim2': "",
                    'trend_day0': oneri_trend_day[0].item(),
                    'trend_day1': "",
                    'trend_day2': "",

                }
                return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 0:
                for i in range(3):
                    """oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])"""
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                    dict = {
                        'tahmim': predict_egitim,
                        'views0': "",
                        'views1': "",
                        'likes0': "",
                        'likes1': "",
                        'dislike0': "",
                        'dislike1': "",
                        'tags1': "No suggestions found",
                        'tags2': "No suggestions found",

                        'tags3': "No suggestions found",
                        'views2': "",
                        'likes2': "",
                        'dislike2': "",
                        'paylasim0': "",
                        'paylasim1': "",
                        'paylasim2': "",
                        'trend_day0': "",
                        'trend_day1': "",
                        'trend_day2': "",

                    }
                    return render(request, 'pages/about.html', dict)















    if ilgi_alani=="music":
        kontrol = 1
        # makine öğrenmesi:
        izlenme_say = music.iloc[:, 6:7]
        abone_say = music.iloc[:, -1]
        abone_say = abone_say.astype('int64')
        # print("sub count",sub_count)
        # print("izlneme",izlneme)
        # print("izlenme eksik veri indexi",a)
        # print("izlenme eksik veri",izlneme.isna().sum())
        # print("abone eksik veri indexi", a)
        # print("abone eksik veri", sub_count.isna().sum())
        x_train, x_test, y_train, y_test = train_test_split(abone_say, izlenme_say, test_size=0.33,
                                                            random_state=0, shuffle=True)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        x_test = x_test.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        tahmin = lr.predict(x_test)

        # ------------------------decision tree
        from sklearn.tree import DecisionTreeRegressor
        X = izlenme_say.values
        Y = abone_say.values
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        r_dt = DecisionTreeRegressor(random_state=0)
        r_dt.fit(Y, X)
        abone_alt_sinir = abone_say.min()
        abone_ust_sinir = abone_say.max()
        izlenme_alt_sinir = izlenme_say.min()
        izlenme_alt_sinir = izlenme_alt_sinir.item()
        izlenme_ust_sinir = izlenme_say.max()
        izlenme_ust_sinir = izlenme_ust_sinir.item()
        print("izlenme:\nust sınır", izlenme_ust_sinir)
        print("alt sınır", izlenme_alt_sinir)
        print("abone:\nust sınır", abone_ust_sinir)
        print("alt sınır", abone_alt_sinir.item())
        if kullanici_sub_say < abone_ust_sinir:
            predict_egitim = r_dt.predict([[kullanici_sub_say]])
            predict_egitim = int(predict_egitim)
        else:
            predict_egitim = lr.predict([[kullanici_sub_say]])
            predict_egitim = int(predict_egitim)
            if predict_egitim < 0:
                predict_egitim = predict_egitim * (-1)
        print("tahmin değeri:", predict_egitim)

        if predict_egitim > izlenme_alt_sinir and predict_egitim < izlenme_ust_sinir:
            fikir_alt_sinir = predict_egitim - 5000
            fikir_ust_sinir = predict_egitim + 5000
            x_ideal = music[music['views'].between(fikir_alt_sinir, fikir_ust_sinir)]
            print("x ideal=", x_ideal['title'])
            """oneri_title0 = x_ideal['title'].iloc[[0]]
            oneri_title1 = x_ideal['title'].iloc[[1]]
            oneri_views0 = x_ideal['views'].iloc[[0]]
            oneri_views1 = x_ideal['views'].iloc[[1]]
            oneri_likes0 = x_ideal['likes'].iloc[[0]]
            oneri_likes1 = x_ideal['likes'].iloc[[1]]
            oneri_dislikes0 = x_ideal['dislikes'].iloc[[0]]
            oneri_dislikes1 = x_ideal['dislikes'].iloc[[1]]"""
            oneri_title = []
            oneri_views = []
            oneri_likes = []
            oneri_dislikes = []
            oneri_paylasim_zamani = []
            oneri_trend_day=[]
            if len(x_ideal['title']) >= 3:
                for i in range(3):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    print("öneri views", type(oneri_views))
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                dict = {
                    'tahmim': predict_egitim,
                    'views0': oneri_views[0].item(),
                    'views1': oneri_views[1].item(),
                    'likes0': oneri_likes[0].item(),
                    'likes1': oneri_likes[1].item(),
                    'dislike0': oneri_dislikes[0].item(),
                    'dislike1': oneri_dislikes[1].item(),
                    'tags1': oneri_title[0].item(),
                    'tags2': oneri_title[1].item(),
                    'tags3': oneri_title[2].item(),
                    'views2': oneri_views[2].item(),
                    'likes2': oneri_likes[2].item(),
                    'dislike2': oneri_dislikes[2].item(),
                    'paylasim0': oneri_paylasim_zamani[0].item(),
                    'paylasim1': oneri_paylasim_zamani[1].item(),
                    'paylasim2': oneri_paylasim_zamani[2].item(),
                    'trend_day0': oneri_trend_day[0].item(),
                    'trend_day1': oneri_trend_day[1].item(),
                    'trend_day2': oneri_trend_day[2].item(),

                }
                return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 2:
                for i in range(2):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                dict = {
                    'tahmim': predict_egitim,
                    'views0': oneri_views[0].item(),
                    'views1': oneri_views[1].item(),
                    'likes0': oneri_likes[0].item(),
                    'likes1': oneri_likes[1].item(),
                    'dislike0': oneri_dislikes[0].item(),
                    'dislike1': oneri_dislikes[1].item(),
                    'tags1': oneri_title[0].item(),
                    'tags2': oneri_title[1].item(),
                    'paylasim0': oneri_paylasim_zamani[0].item(),
                    'paylasim1': oneri_paylasim_zamani[1].item(),
                    'tags3': "",
                    'views2': "",
                    'likes2': "",
                    'dislike2': "",
                    'paylasim2': "",
                    'trend_day0': oneri_trend_day[0].item(),
                    'trend_day1': oneri_trend_day[1].item(),
                    'trend_day2': "",

                }
                return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 1:
                for i in range(1):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                dict = {
                    'tahmim': predict_egitim,
                    'views0': oneri_views[0].item(),
                    'views1': "",
                    'likes0': oneri_likes[0].item(),
                    'likes1': "",
                    'dislike0': oneri_dislikes[0].item(),
                    'dislike1': "",
                    'tags1': oneri_title[0].item(),
                    'tags2': "",
                    'tags3': "",
                    'views2': "",
                    'likes2': "",
                    'dislike2': "",
                    'paylasim0': oneri_paylasim_zamani[0].item(),
                    'paylasim1': "",
                    'paylasim2': "",
                    'trend_day0': oneri_trend_day[0].item(),
                    'trend_day1': "",
                    'trend_day2': "",

                }
                return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 0:
                for i in range(3):
                    """oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])"""
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                    dict = {
                        'tahmim': predict_egitim,
                        'views0': "",
                        'views1': "",
                        'likes0': "",
                        'likes1': "",
                        'dislike0': "",
                        'dislike1': "",
                        'tags1': "No suggestions found",
                        'tags2': "No suggestions found",

                        'tags3': "No suggestions found",
                        'views2': "",
                        'likes2': "",
                        'dislike2': "",
                        'paylasim0': "",
                        'paylasim1': "",
                        'paylasim2': "",
                        'trend_day0': "",
                        'trend_day1': "",
                        'trend_day2': "",

                    }
                    return render(request, 'pages/about.html', dict)








    if ilgi_alani=="oyun":
        kontrol = 1
        # makine öğrenmesi:
        izlenme_say = oyun.iloc[:, 6:7]
        abone_say = oyun.iloc[:, -1]
        abone_say = abone_say.astype('int64')
        # print("sub count",sub_count)
        # print("izlneme",izlneme)
        # print("izlenme eksik veri indexi",a)
        # print("izlenme eksik veri",izlneme.isna().sum())
        # print("abone eksik veri indexi", a)
        # print("abone eksik veri", sub_count.isna().sum())
        x_train, x_test, y_train, y_test = train_test_split(abone_say, izlenme_say, test_size=0.33,
                                                            random_state=0, shuffle=True)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        x_test = x_test.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        tahmin = lr.predict(x_test)

        # ------------------------decision tree
        from sklearn.tree import DecisionTreeRegressor
        X = izlenme_say.values
        Y = abone_say.values
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        r_dt = DecisionTreeRegressor(random_state=0)
        r_dt.fit(Y, X)
        abone_alt_sinir = abone_say.min()
        abone_ust_sinir = abone_say.max()
        izlenme_alt_sinir = izlenme_say.min()
        izlenme_alt_sinir = izlenme_alt_sinir.item()
        izlenme_ust_sinir = izlenme_say.max()
        izlenme_ust_sinir = izlenme_ust_sinir.item()
        print("izlenme:\nust sınır", izlenme_ust_sinir)
        print("alt sınır", izlenme_alt_sinir)
        print("abone:\nust sınır", abone_ust_sinir)
        print("alt sınır", abone_alt_sinir.item())
        if kullanici_sub_say < abone_ust_sinir:
            predict_egitim = r_dt.predict([[kullanici_sub_say]])
            predict_egitim = int(predict_egitim)
        else:
            predict_egitim = lr.predict([[kullanici_sub_say]])
            predict_egitim = int(predict_egitim)
            if predict_egitim < 0:
                predict_egitim = predict_egitim * (-1)
        print("tahmin değeri:", predict_egitim)

        if predict_egitim > izlenme_alt_sinir and predict_egitim < izlenme_ust_sinir:
            fikir_alt_sinir = predict_egitim - 5000
            fikir_ust_sinir = predict_egitim + 5000
            x_ideal = oyun[oyun['views'].between(fikir_alt_sinir, fikir_ust_sinir)]
            print("x ideal=", x_ideal['title'])
            """oneri_title0 = x_ideal['title'].iloc[[0]]
            oneri_title1 = x_ideal['title'].iloc[[1]]
            oneri_views0 = x_ideal['views'].iloc[[0]]
            oneri_views1 = x_ideal['views'].iloc[[1]]
            oneri_likes0 = x_ideal['likes'].iloc[[0]]
            oneri_likes1 = x_ideal['likes'].iloc[[1]]
            oneri_dislikes0 = x_ideal['dislikes'].iloc[[0]]
            oneri_dislikes1 = x_ideal['dislikes'].iloc[[1]]"""
            oneri_title = []
            oneri_views = []
            oneri_likes = []
            oneri_dislikes = []
            oneri_paylasim_zamani = []
            oneri_trend_day=[]
            if len(x_ideal['title']) >= 3:
                for i in range(3):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    print("öneri views", type(oneri_views))
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                dict = {
                    'tahmim': predict_egitim,
                    'views0': oneri_views[0].item(),
                    'views1': oneri_views[1].item(),
                    'likes0': oneri_likes[0].item(),
                    'likes1': oneri_likes[1].item(),
                    'dislike0': oneri_dislikes[0].item(),
                    'dislike1': oneri_dislikes[1].item(),
                    'tags1': oneri_title[0].item(),
                    'tags2': oneri_title[1].item(),
                    'tags3': oneri_title[2].item(),
                    'views2': oneri_views[2].item(),
                    'likes2': oneri_likes[2].item(),
                    'dislike2': oneri_dislikes[2].item(),
                    'paylasim0': oneri_paylasim_zamani[0].item(),
                    'paylasim1': oneri_paylasim_zamani[1].item(),
                    'paylasim2': oneri_paylasim_zamani[2].item(),
                    'trend_day0': oneri_trend_day[0].item(),
                    'trend_day1': oneri_trend_day[1].item(),
                    'trend_day2': oneri_trend_day[2].item(),

                }
                return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 2:
                for i in range(2):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                dict = {
                    'tahmim': predict_egitim,
                    'views0': oneri_views[0].item(),
                    'views1': oneri_views[1].item(),
                    'likes0': oneri_likes[0].item(),
                    'likes1': oneri_likes[1].item(),
                    'dislike0': oneri_dislikes[0].item(),
                    'dislike1': oneri_dislikes[1].item(),
                    'tags1': oneri_title[0].item(),
                    'tags2': oneri_title[1].item(),
                    'paylasim0': oneri_paylasim_zamani[0].item(),
                    'paylasim1': oneri_paylasim_zamani[1].item(),
                    'tags3': "",
                    'views2': "",
                    'likes2': "",
                    'dislike2': "",
                    'paylasim2': "",
                    'trend_day0': oneri_trend_day[0].item(),
                    'trend_day1': oneri_trend_day[1].item(),
                    'trend_day2': "",

                }
                return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 1:
                for i in range(1):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                dict = {
                    'tahmim': predict_egitim,
                    'views0': oneri_views[0].item(),
                    'views1': "",
                    'likes0': oneri_likes[0].item(),
                    'likes1': "",
                    'dislike0': oneri_dislikes[0].item(),
                    'dislike1': "",
                    'tags1': oneri_title[0].item(),
                    'tags2': "",
                    'tags3': "",
                    'views2': "",
                    'likes2': "",
                    'dislike2': "",
                    'paylasim0': oneri_paylasim_zamani[0].item(),
                    'paylasim1': "",
                    'paylasim2': "",
                    'trend_day0': oneri_trend_day[0].item(),
                    'trend_day1': "",
                    'trend_day2': "",

                }
                return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 0:
                for i in range(3):
                    """oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])"""
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                    dict = {
                        'tahmim': predict_egitim,
                        'views0': "",
                        'views1': "",
                        'likes0': "",
                        'likes1': "",
                        'dislike0': "",
                        'dislike1': "",
                        'tags1': "No suggestions found",
                        'tags2': "No suggestions found",

                        'tags3': "No suggestions found",
                        'views2': "",
                        'likes2': "",
                        'dislike2': "",
                        'paylasim0': "",
                        'paylasim1': "",
                        'paylasim2': "",
                        'trend_day0': "",
                        'trend_day1': "",
                        'trend_day2': "",

                    }
                    return render(request, 'pages/about.html', dict)










    if ilgi_alani=="siyasi":
        kontrol = 1
        # makine öğrenmesi:
        izlenme_say = siyasi.iloc[:, 6:7]
        abone_say = siyasi.iloc[:, -1]
        abone_say = abone_say.astype('int64')
        # print("sub count",sub_count)
        # print("izlneme",izlneme)
        # print("izlenme eksik veri indexi",a)
        # print("izlenme eksik veri",izlneme.isna().sum())
        # print("abone eksik veri indexi", a)
        # print("abone eksik veri", sub_count.isna().sum())
        x_train, x_test, y_train, y_test = train_test_split(abone_say, izlenme_say, test_size=0.33,
                                                            random_state=0, shuffle=True)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        x_test = x_test.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        tahmin = lr.predict(x_test)

        # ------------------------decision tree
        from sklearn.tree import DecisionTreeRegressor
        X = izlenme_say.values
        Y = abone_say.values
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        r_dt = DecisionTreeRegressor(random_state=0)
        r_dt.fit(Y, X)
        abone_alt_sinir = abone_say.min()
        abone_ust_sinir = abone_say.max()
        izlenme_alt_sinir = izlenme_say.min()
        izlenme_alt_sinir = izlenme_alt_sinir.item()
        izlenme_ust_sinir = izlenme_say.max()
        izlenme_ust_sinir = izlenme_ust_sinir.item()
        print("izlenme:\nust sınır", izlenme_ust_sinir)
        print("alt sınır", izlenme_alt_sinir)
        print("abone:\nust sınır", abone_ust_sinir)
        print("alt sınır", abone_alt_sinir.item())
        if kullanici_sub_say < abone_ust_sinir:
            predict_egitim = r_dt.predict([[kullanici_sub_say]])
            predict_egitim = int(predict_egitim)
        else:
            predict_egitim = lr.predict([[kullanici_sub_say]])
            predict_egitim = int(predict_egitim)
            if predict_egitim < 0:
                predict_egitim = predict_egitim * (-1)
        print("tahmin değeri:", predict_egitim)

        if predict_egitim > izlenme_alt_sinir and predict_egitim < izlenme_ust_sinir:
            fikir_alt_sinir = predict_egitim - 5000
            fikir_ust_sinir = predict_egitim + 5000
            x_ideal = siyasi[siyasi['views'].between(fikir_alt_sinir, fikir_ust_sinir)]
            print("x ideal=", x_ideal['title'])
            """oneri_title0 = x_ideal['title'].iloc[[0]]
            #oneri_title1 = x_ideal['title'].iloc[[1]]
            oneri_views0 = x_ideal['views'].iloc[[0]]
            #oneri_views1 = x_ideal['views'].iloc[[1]]
            oneri_likes0 = x_ideal['likes'].iloc[[0]]
            #oneri_likes1 = x_ideal['likes'].iloc[[1]]
            oneri_dislikes0 = x_ideal['dislikes'].iloc[[0]]
            #oneri_dislikes1 = x_ideal['dislikes'].iloc[[1]]"""
            oneri_title = []
            oneri_views = []
            oneri_likes = []
            oneri_dislikes = []
            oneri_paylasim_zamani = []
            oneri_trend_day=[]
            if len(x_ideal['title']) >= 3:
                for i in range(3):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    print("öneri views", type(oneri_views))
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                dict = {
                    'tahmim': predict_egitim,
                    'views0': oneri_views[0].item(),
                    'views1': oneri_views[1].item(),
                    'likes0': oneri_likes[0].item(),
                    'likes1': oneri_likes[1].item(),
                    'dislike0': oneri_dislikes[0].item(),
                    'dislike1': oneri_dislikes[1].item(),
                    'tags1': oneri_title[0].item(),
                    'tags2': oneri_title[1].item(),
                    'tags3': oneri_title[2].item(),
                    'views2': oneri_views[2].item(),
                    'likes2': oneri_likes[2].item(),
                    'dislike2': oneri_dislikes[2].item(),
                    'paylasim0': oneri_paylasim_zamani[0].item(),
                    'paylasim1': oneri_paylasim_zamani[1].item(),
                    'paylasim2': oneri_paylasim_zamani[2].item(),
                    'trend_day0': oneri_trend_day[0].item(),
                    'trend_day1': oneri_trend_day[1].item(),
                    'trend_day2': oneri_trend_day[2].item(),

                }
                return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 2:
                for i in range(2):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                dict = {
                    'tahmim': predict_egitim,
                    'views0': oneri_views[0].item(),
                    'views1': oneri_views[1].item(),
                    'likes0': oneri_likes[0].item(),
                    'likes1': oneri_likes[1].item(),
                    'dislike0': oneri_dislikes[0].item(),
                    'dislike1': oneri_dislikes[1].item(),
                    'tags1': oneri_title[0].item(),
                    'tags2': oneri_title[1].item(),
                    'paylasim0': oneri_paylasim_zamani[0].item(),
                    'paylasim1': oneri_paylasim_zamani[1].item(),
                    'tags3': "",
                    'views2': "",
                    'likes2': "",
                    'dislike2': "",
                    'paylasim2': "",
                    'trend_day0': oneri_trend_day[0].item(),
                    'trend_day1': oneri_trend_day[1].item(),
                    'trend_day2': "",

                }
                return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 1:
                for i in range(1):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                dict = {
                    'tahmim': predict_egitim,
                    'views0': oneri_views[0].item(),
                    'views1': "",
                    'likes0': oneri_likes[0].item(),
                    'likes1': "",
                    'dislike0': oneri_dislikes[0].item(),
                    'dislike1': "",
                    'tags1': oneri_title[0].item(),
                    'tags2': "",
                    'tags3': "",
                    'views2': "",
                    'likes2': "",
                    'dislike2': "",
                    'paylasim0': oneri_paylasim_zamani[0].item(),
                    'paylasim1': "",
                    'paylasim2': "",
                    'trend_day0': oneri_trend_day[0].item(),
                    'trend_day1': "",
                    'trend_day2': "",

                }
                return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 0:
                for i in range(3):
                    """oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])"""
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                    dict = {
                        'tahmim': predict_egitim,
                        'views0': "",
                        'views1': "",
                        'likes0': "",
                        'likes1': "",
                        'dislike0': "",
                        'dislike1': "",
                        'tags1': "No suggestions found",
                        'tags2': "No suggestions found",

                        'tags3': "No suggestions found",
                        'views2': "",
                        'likes2': "",
                        'dislike2': "",
                        'paylasim0': "",
                        'paylasim1': "",
                        'paylasim2': "",
                        'trend_day0': "",
                        'trend_day1': "",
                        'trend_day2': "",

                    }
                    return render(request, 'pages/about.html', dict)










    if ilgi_alani=="spor":
        kontrol = 1
        # makine öğrenmesi:
        izlenme_say = spor.iloc[:, 6:7]
        abone_say = spor.iloc[:, -1]
        abone_say = abone_say.astype('int64')
        # print("sub count",sub_count)
        # print("izlneme",izlneme)
        # print("izlenme eksik veri indexi",a)
        # print("izlenme eksik veri",izlneme.isna().sum())
        # print("abone eksik veri indexi", a)
        # print("abone eksik veri", sub_count.isna().sum())
        x_train, x_test, y_train, y_test = train_test_split(abone_say, izlenme_say, test_size=0.33,
                                                            random_state=0, shuffle=True)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        x_test = x_test.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        tahmin = lr.predict(x_test)

        # ------------------------decision tree
        from sklearn.tree import DecisionTreeRegressor
        X = izlenme_say.values
        Y = abone_say.values
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        r_dt = DecisionTreeRegressor(random_state=0)
        r_dt.fit(Y, X)
        abone_alt_sinir = abone_say.min()
        abone_ust_sinir = abone_say.max()
        izlenme_alt_sinir = izlenme_say.min()
        izlenme_alt_sinir = izlenme_alt_sinir.item()
        izlenme_ust_sinir = izlenme_say.max()
        izlenme_ust_sinir = izlenme_ust_sinir.item()
        print("izlenme:\nust sınır", izlenme_ust_sinir)
        print("alt sınır", izlenme_alt_sinir)
        print("abone:\nust sınır", abone_ust_sinir)
        print("alt sınır", abone_alt_sinir.item())
        if kullanici_sub_say < abone_ust_sinir:
            predict_egitim = r_dt.predict([[kullanici_sub_say]])
            predict_egitim = int(predict_egitim)
        else:
            predict_egitim = lr.predict([[kullanici_sub_say]])
            predict_egitim = int(predict_egitim)
            if predict_egitim < 0:
                predict_egitim = predict_egitim * (-1)
        print("tahmin değeri:", predict_egitim)

        if predict_egitim > izlenme_alt_sinir and predict_egitim < izlenme_ust_sinir:
            fikir_alt_sinir = predict_egitim - 5000
            fikir_ust_sinir = predict_egitim + 5000
            x_ideal = spor[spor['views'].between(fikir_alt_sinir, fikir_ust_sinir)]
            print("x ideal=", x_ideal['title'])
            """oneri_title0 = x_ideal['title'].iloc[[0]]
            oneri_title1 = x_ideal['title'].iloc[[1]]
            oneri_views0 = x_ideal['views'].iloc[[0]]
            oneri_views1 = x_ideal['views'].iloc[[1]]
            oneri_likes0 = x_ideal['likes'].iloc[[0]]
            oneri_likes1 = x_ideal['likes'].iloc[[1]]
            oneri_dislikes0 = x_ideal['dislikes'].iloc[[0]]
            oneri_dislikes1 = x_ideal['dislikes'].iloc[[1]]"""
            oneri_title = []
            oneri_views = []
            oneri_likes = []
            oneri_dislikes = []
            oneri_paylasim_zamani = []
            oneri_trend_day=[]
            if len(x_ideal['title']) >= 3:
                for i in range(3):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    print("öneri views", type(oneri_views))
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                dict = {
                    'tahmim': predict_egitim,
                    'views0': oneri_views[0].item(),
                    'views1': oneri_views[1].item(),
                    'likes0': oneri_likes[0].item(),
                    'likes1': oneri_likes[1].item(),
                    'dislike0': oneri_dislikes[0].item(),
                    'dislike1': oneri_dislikes[1].item(),
                    'tags1': oneri_title[0].item(),
                    'tags2': oneri_title[1].item(),
                    'tags3': oneri_title[2].item(),
                    'views2': oneri_views[2].item(),
                    'likes2': oneri_likes[2].item(),
                    'dislike2': oneri_dislikes[2].item(),
                    'paylasim0': oneri_paylasim_zamani[0].item(),
                    'paylasim1': oneri_paylasim_zamani[1].item(),
                    'paylasim2': oneri_paylasim_zamani[2].item(),
                    'trend_day0': oneri_trend_day[0].item(),
                    'trend_day1': oneri_trend_day[1].item(),
                    'trend_day2': oneri_trend_day[2].item(),

                }
                return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 2:
                for i in range(2):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                dict = {
                    'tahmim': predict_egitim,
                    'views0': oneri_views[0].item(),
                    'views1': oneri_views[1].item(),
                    'likes0': oneri_likes[0].item(),
                    'likes1': oneri_likes[1].item(),
                    'dislike0': oneri_dislikes[0].item(),
                    'dislike1': oneri_dislikes[1].item(),
                    'tags1': oneri_title[0].item(),
                    'tags2': oneri_title[1].item(),
                    'paylasim0': oneri_paylasim_zamani[0].item(),
                    'paylasim1': oneri_paylasim_zamani[1].item(),
                    'tags3': "",
                    'views2': "",
                    'likes2': "",
                    'dislike2': "",
                    'paylasim2': "",
                    'trend_day0': oneri_trend_day[0].item(),
                    'trend_day1': oneri_trend_day[1].item(),
                    'trend_day2': "",

                }
                return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 1:
                for i in range(1):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                dict = {
                    'tahmim': predict_egitim,
                    'views0': oneri_views[0].item(),
                    'views1': "",
                    'likes0': oneri_likes[0].item(),
                    'likes1': "",
                    'dislike0': oneri_dislikes[0].item(),
                    'dislike1': "",
                    'tags1': oneri_title[0].item(),
                    'tags2': "",
                    'tags3': "",
                    'views2': "",
                    'likes2': "",
                    'dislike2': "",
                    'paylasim0': oneri_paylasim_zamani[0].item(),
                    'paylasim1': "",
                    'paylasim2': "",
                    'trend_day0': oneri_trend_day[0].item(),
                    'trend_day1': "",
                    'trend_day2': "",

                }
                return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 0:
                for i in range(3):
                    """oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])"""
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                    dict = {
                        'tahmim': predict_egitim,
                        'views0': "",
                        'views1': "",
                        'likes0': "",
                        'likes1': "",
                        'dislike0': "",
                        'dislike1': "",
                        'tags1': "No suggestions found",
                        'tags2': "No suggestions found",

                        'tags3': "No suggestions found",
                        'views2': "",
                        'likes2': "",
                        'dislike2': "",
                        'paylasim0': "",
                        'paylasim1': "",
                        'paylasim2': "",
                        'trend_day0': "",
                        'trend_day1': "",
                        'trend_day2': "",

                    }
                    return render(request, 'pages/about.html', dict)










    if ilgi_alani=="yemek":
        kontrol = 1
        # makine öğrenmesi:
        izlenme_say = yemek.iloc[:, 6:7]
        abone_say = yemek.iloc[:, -1]
        abone_say = abone_say.astype('int64')
        # print("sub count",sub_count)
        # print("izlneme",izlneme)
        # print("izlenme eksik veri indexi",a)
        # print("izlenme eksik veri",izlneme.isna().sum())
        # print("abone eksik veri indexi", a)
        # print("abone eksik veri", sub_count.isna().sum())
        x_train, x_test, y_train, y_test = train_test_split(abone_say, izlenme_say, test_size=0.33,
                                                            random_state=0, shuffle=True)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        x_test = x_test.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        tahmin = lr.predict(x_test)

        # ------------------------decision tree
        from sklearn.tree import DecisionTreeRegressor
        X = izlenme_say.values
        Y = abone_say.values
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        r_dt = DecisionTreeRegressor(random_state=0)
        r_dt.fit(Y, X)
        abone_alt_sinir = abone_say.min()
        abone_ust_sinir = abone_say.max()
        izlenme_alt_sinir = izlenme_say.min()
        izlenme_alt_sinir = izlenme_alt_sinir.item()
        izlenme_ust_sinir = izlenme_say.max()
        izlenme_ust_sinir = izlenme_ust_sinir.item()
        print("izlenme:\nust sınır", izlenme_ust_sinir)
        print("alt sınır", izlenme_alt_sinir)
        print("abone:\nust sınır", abone_ust_sinir)
        print("alt sınır", abone_alt_sinir.item())
        if kullanici_sub_say < abone_ust_sinir:
            predict_egitim = r_dt.predict([[kullanici_sub_say]])
            predict_egitim = int(predict_egitim)
        else:
            predict_egitim = lr.predict([[kullanici_sub_say]])
            predict_egitim = int(predict_egitim)
            if predict_egitim < 0:
                predict_egitim = predict_egitim * (-1)
        print("tahmin değeri:", predict_egitim)
    #31.35+24,76
        if predict_egitim > izlenme_alt_sinir and predict_egitim < izlenme_ust_sinir:
            fikir_alt_sinir = predict_egitim - 5000
            fikir_ust_sinir = predict_egitim + 5000
            x_ideal = yemek[yemek['views'].between(fikir_alt_sinir, fikir_ust_sinir)]
            print("x ideal=", x_ideal['title'])
            """oneri_title0 = x_ideal['title'].iloc[[0]]
            oneri_title1 = x_ideal['title'].iloc[[1]]
            oneri_views0 = x_ideal['views'].iloc[[0]]
            oneri_views1 = x_ideal['views'].iloc[[1]]
            oneri_likes0 = x_ideal['likes'].iloc[[0]]
            oneri_likes1 = x_ideal['likes'].iloc[[1]]
            oneri_dislikes0 = x_ideal['dislikes'].iloc[[0]]
            oneri_dislikes1 = x_ideal['dislikes'].iloc[[1]]"""
            oneri_title = []
            oneri_views = []
            oneri_likes = []
            oneri_dislikes = []
            oneri_paylasim_zamani = []
            oneri_trend_day=[]
            if len(x_ideal['title']) >= 3:
                for i in range(3):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    saz=(" word sports",x_ideal['tags'].iloc[[1]])
                    print(saz)
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                dict = {
                    'tahmim': predict_egitim,
                    'views0': oneri_views[0].item(),
                    'views1': oneri_views[1].item(),
                    'likes0': oneri_likes[0].item(),
                    'likes1': oneri_likes[1].item(),
                    'dislike0': oneri_dislikes[0].item(),
                    'dislike1': oneri_dislikes[1].item(),
                    'tags1': oneri_title[0].item(),
                    'tags2': oneri_title[1].item(),
                    'tags3': oneri_title[2].item(),
                    'views2': oneri_views[2].item(),
                    'likes2': oneri_likes[2].item(),
                    'dislike2': oneri_dislikes[2].item(),
                    'paylasim0': oneri_paylasim_zamani[0].item(),
                    'paylasim1': oneri_paylasim_zamani[1].item(),
                    'paylasim2': oneri_paylasim_zamani[2].item(),
                    'trend_day0': oneri_trend_day[0].item(),
                    'trend_day1': oneri_trend_day[1].item(),
                    'trend_day2': oneri_trend_day[2].item(),

                }
                return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 2:
                for i in range(2):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                dict = {
                    'tahmim': predict_egitim,
                    'views0': oneri_views[0].item(),
                    'views1': oneri_views[1].item(),
                    'likes0': oneri_likes[0].item(),
                    'likes1': oneri_likes[1].item(),
                    'dislike0': oneri_dislikes[0].item(),
                    'dislike1': oneri_dislikes[1].item(),
                    'tags1': oneri_title[0].item(),
                    'tags2': oneri_title[1].item(),
                    'paylasim0': oneri_paylasim_zamani[0].item(),
                    'paylasim1': oneri_paylasim_zamani[1].item(),
                    'tags3': "",
                    'views2': "",
                    'likes2': "",
                    'dislike2': "",
                    'paylasim2': "",
                    'trend_day0': oneri_trend_day[0].item(),
                    'trend_day1': oneri_trend_day[1].item(),
                    'trend_day2': "",

                }
                return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 1:
                for i in range(1):
                    oneri_paylasim_zamani.append(x_ideal['publish_date'].iloc[[i]])
                    oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])
                    oneri_trend_day.append(x_ideal['trend_day_count'].iloc[[i]])
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                dict = {
                    'tahmim': predict_egitim,
                    'views0': oneri_views[0].item(),
                    'views1': "",
                    'likes0': oneri_likes[0].item(),
                    'likes1': "",
                    'dislike0': oneri_dislikes[0].item(),
                    'dislike1': "",
                    'tags1': oneri_title[0].item(),
                    'tags2': "",
                    'tags3': "",
                    'views2': "",
                    'likes2': "",
                    'dislike2': "",
                    'paylasim0': oneri_paylasim_zamani[0].item(),
                    'paylasim1': "",
                    'paylasim2': "",
                    'trend_day0': oneri_trend_day[0].item(),
                    'trend_day1': "",
                    'trend_day2': "",

                }
                return render(request, 'pages/about.html', dict)
            if len(x_ideal['title']) == 0:
                for i in range(3):
                    """oneri_title.append(x_ideal['title'].iloc[[i]])
                    oneri_views.append(x_ideal['views'].iloc[[i]])
                    oneri_likes.append(x_ideal['likes'].iloc[[i]])
                    oneri_dislikes.append(x_ideal['dislikes'].iloc[[i]])"""
                    """oneri_title= x_ideal['title'].iloc[[1]]
                    oneri_views= x_ideal['views'].iloc[[1]]
                    oneri_likes= x_ideal['likes'].iloc[[1]]
                    oneri_dislikes= x_ideal['dislikes'].iloc[[1]]"""
                    dict = {
                        'tahmim': predict_egitim,
                        'views0': "",
                        'views1': "",
                        'likes0': "",
                        'likes1': "",
                        'dislike0': "",
                        'dislike1': "",
                        'tags1': "No suggestions found",
                        'tags2': "No suggestions found",

                        'tags3': "No suggestions found",
                        'views2': "",
                        'likes2': "",
                        'dislike2': "",
                        'paylasim0': "",
                        'paylasim1': "",
                        'paylasim2': "",
                        'trend_day0': "",
                        'trend_day1': "",
                        'trend_day2': "",

                    }
                    return render(request, 'pages/about.html', dict)












    """if kontrol==0:
        veriler = pd.read_csv('pages/static/channels.csv')
        print(veriler['subs_count'])
        izlenme_veri = pd.read_csv('pages/static/newlike.csv')
        ilgi_alanlari_csv = pd.read_csv('pages/static/tum_veriler.csv')
        # veriler['subs_count'] = veriler['subs_count'].str.replace(',', '').astype(np.int64)
        # veriler['video views'] = veriler['video views'].str.replace(',', '').astype(np.int64)
        # veriler['video count'] = veriler['video count'].str.replace(',', '').astype(np.int64)
        # sıfır olan verileri medyan ile düzenledik
        veriler['video_count'] = veriler['video_count'].replace(0, veriler['video_count'].median())
        veriler['view_count'] = veriler['view_count'].replace(0, veriler['view_count'].median())
        arrayy = []

        for i in range(len(veriler['view_count'])):
            veri = int(veriler['view_count'][i] / veriler['video_count'][i])
            arrayy.append(veri)

        np_array = np.asarray(arrayy)
        veriler['oran'] = np_array
        # abone sayısından izlenme tahmin
        sub_count = veriler.iloc[:, 6:7]
        veriler_oran = veriler.iloc[:, 13]
        x_train, x_test, y_train, y_test = train_test_split(sub_count, veriler_oran, test_size=0.50,
                                                            random_state=0, shuffle=True)
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        tahmin = lr.predict(x_test)
        # ------------------------decision tree
        from sklearn.tree import DecisionTreeRegressor
        X = sub_count.values
        Y = veriler_oran.values
        r_dt = DecisionTreeRegressor(random_state=0)
        r_dt.fit(X, Y)
        x = kullanici_sub_say
        az = np.min(sub_count)
        cok = np.max(sub_count)
        print("az", az.item())
        print("cok", cok.item())
        print("[[x]]=", [[x]])
        if x < az.item() or x > cok.item():
            print("dogrusal reg", int(lr.predict([[x]])))
            predict_degisken = int(lr.predict([[kullanici_sub_say]]))
            if predict_degisken < 0:
                predict_degisken = predict_degisken * (-1)
        else:
            print("int(r_dt.predict([[kullanici_sub_say]]", int(r_dt.predict([[kullanici_sub_say]])))
            predict_degisken = r_dt.predict([[kullanici_sub_say]])
        print("son predick degisken=", predict_degisken)
        alt = predict_degisken - 1000
        ust = predict_degisken + 1000
        ilgi_alanlari_csv['izleme'] = ilgi_alanlari_csv['views'].astype(str).str.replace(',', '').astype(np.int64)
        print(ilgi_alanlari_csv['izleme'])
        max_izlemne = ilgi_alanlari_csv['izleme'].max()
        print(max_izlemne)

        x_ideal = ilgi_alanlari_csv[ilgi_alanlari_csv['izleme'].between(alt.item(), ust.item())]
        print("x_ideal=", x_ideal)

        # where((predict_degisken+100)>izlenme_veri['views'](predict_degisken-100)<izlenme_veri['views'])

        sozcuk1 = x_ideal['tags'].values.tolist()
        print("sozcuk1=", sozcuk1)
        ilgili = []
        for i in range(len(sozcuk1)):
            if sozcuk1[i] == ilgi_alani:
                ilgili.append(sozcuk1[i])
        if len(ilgili) == 0:
            ilgili.append("İlgi alanınızla ilgili bir etiket bulunamadı")

        # 17462656
        dict = {
            'views': int(predict_degisken),
            'likes': 2,
            'dislike': 2,
            'tags1': 2,
            'tags2': 2,
            'tags3': 2,

        }

        return render(request, 'pages/about.html', dict)"""





"""veriler = pd.read_csv('pages/static/channels.csv')
    print(veriler['subs_count'])
    izlenme_veri=pd.read_csv('pages/static/newlike.csv')
    ilgi_alanlari_csv=pd.read_csv('pages/static/tum_veriler.csv')
    #veriler['subs_count'] = veriler['subs_count'].str.replace(',', '').astype(np.int64)
    #veriler['video views'] = veriler['video views'].str.replace(',', '').astype(np.int64)
    #veriler['video count'] = veriler['video count'].str.replace(',', '').astype(np.int64)
    # sıfır olan verileri medyan ile düzenledik
    veriler['video_count'] = veriler['video_count'].replace(0, veriler['video_count'].median())
    veriler['view_count'] = veriler['view_count'].replace(0, veriler['view_count'].median())
    arrayy = []

    for i in range(len(veriler['view_count'])):
        veri = int(veriler['view_count'][i] / veriler['video_count'][i])
        arrayy.append(veri)

    np_array = np.asarray(arrayy)
    veriler['oran'] = np_array
    # abone sayısından izlenme tahmin
    sub_count = veriler.iloc[:, 8:9]
    veriler_oran = veriler.iloc[:, 13]
    x_train, x_test, y_train, y_test = train_test_split(sub_count, veriler_oran, test_size=0.50,
                                                       random_state=0,shuffle=True)
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    tahmin = lr.predict(x_test)
    # ------------------------decision tree
    from sklearn.tree import DecisionTreeRegressor
    X = sub_count.values
    Y = veriler_oran.values
    r_dt = DecisionTreeRegressor(random_state=0)
    r_dt.fit(X, Y)
    x =kullanici_sub_say
    az = np.min(sub_count)
    cok = np.max(sub_count)
    print("az",az.item())
    print("cok",cok.item())
    print("[[x]]=",[[x]])
    if x < az.item() or x > cok.item():
        print("dogrusal reg",int(lr.predict([[x]])))
        predict_degisken = int(lr.predict([[kullanici_sub_say]]))
        if predict_degisken<0:
            predict_degisken=predict_degisken*(-1)
    else:
        print("int(r_dt.predict([[kullanici_sub_say]]",int(r_dt.predict([[kullanici_sub_say]])))
        predict_degisken = r_dt.predict([[kullanici_sub_say]])
    print("son predick degisken=",predict_degisken)
    alt=predict_degisken-1000
    ust=predict_degisken+1000
    ilgi_alanlari_csv['izleme']= ilgi_alanlari_csv['views'].astype(str).str.replace(',', '').astype(np.int64)
    print(ilgi_alanlari_csv['izleme'])
    max_izlemne=ilgi_alanlari_csv['izleme'].max()
    print(max_izlemne)

    x_ideal = ilgi_alanlari_csv[ilgi_alanlari_csv['izleme'].between(alt.item(), ust.item())]
    print("x_ideal=",x_ideal)



    #where((predict_degisken+100)>izlenme_veri['views'](predict_degisken-100)<izlenme_veri['views'])


    sozcuk1=x_ideal['tags'].values.tolist()
    print("sozcuk1=",sozcuk1)
    ilgili=[]
    for i in range(len(sozcuk1)):
        if sozcuk1[i]==ilgi_alani:
            ilgili.append(sozcuk1[i])
    if len(ilgili)==0:
        ilgili.append("İlgi alanınızla ilgili bir etiket bulunamadı")

    #17462656
"""


