import pandas as pd
import seaborn as sns
import numpy as np
pd.set_option("display.max.rows",None)
pd.set_option("display.max.columns",None)
pd.set_option("display.width",500)

df = sns.load_dataset("car_crashes")
df.columns
df.info()

#List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini büyük harfe çeviriniz ve başına NUM ekleyiniz
["NUM_"+col.upper() if df[col].dtype!="O" else col.upper() for col in df.columns] #sadece sayısal kolonlara num getirdik diğerini sadece büyüttük

#List Comprehension yapısı kullanarak car_crashes verisinde isminde "no" barındırmayan değişkenlerin isimlerinin sonuna "FLAG" yazınız
[col.upper()+"_FLAG" if "no" not in col else col.upper() for col in df.columns]

#List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin isimlerini seçiniz ve yeni bir dataframe oluşturunuz
og_list=["abbrev","no_previous"]

new_cols = [col for col in df.columns if col not in og_list]
df = df[new_cols]
df.columns

#*************************#

#Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
titanic=sns.load_dataset("Titanic")
titanic.columns
titanic.shape

#Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
titanic["sex"].value_counts()

#Her bir sutuna ait unique değerlerin sayısını bulunuz.
titanic.nunique()

#pclass değişkeninin unique değerlerinin sayısını bulunuz
titanic["pclass"].unique()
titanic["pclass"].nunique()

#pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz
titanic[["pclass","parch"]].nunique()

#embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz
titanic["embarked"].dtype
titanic["embarked"] = titanic["embarked"].astype("category")
titanic["embarked"].dtype

#embarked değeri C olanların tüm bilgelerini gösteriniz
titanic[titanic["embarked"]=="C"].head(10)

#embarked değeri S olmayanların tüm bilgelerini gösteriniz.
titanic[titanic["embarked"]!="S"].head(10)

#Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz
titanic[(titanic["age"]<30) & (titanic["sex"]=="female")]

#Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
titanic[(titanic["age"]>70) | (titanic["fare"]>500)]["sex"]

#Her bir değişkendeki boş değerlerin toplamını bulunuz
titanic.isnull().sum()

#who değişkenini dataframe’den çıkarınız.
titanic.drop("who",axis=1,inplace=True)
titanic.head()

#deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
type(titanic["deck"].mode())
titanic["deck"].mode()[0]
titanic["deck"].fillna(titanic["deck"].mode()[0],inplace=True)
titanic["deck"].isnull().sum()

#age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz
titanic["age"].fillna(titanic["age"].median(),inplace=True)
titanic.isnull().sum()

#survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
titanic.groupby(["pclass","sex"]).agg({"survived":["sum","count","mean"]}).reset_index()

#30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın. Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız
def age_30(age):
    if age<30:
        return 1
    else:
        return 0

titanic["age_flag"] = titanic["age"].apply(lambda x: age_30(x))
titanic["age_flag"] = titanic["age"].apply(lambda x:1 if x<30 else 0)


#Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
tips = sns.load_dataset("tips")
tips.info
tips.columns

#Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz
tips.groupby("time").agg({"total_bill":["sum","min","max","mean"]})

#Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz
tips.groupby(["day","time"]).agg({"total_bill":["sum","min","max","mean"]}).reset_index()

#Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz
tips[(tips["time"]=="Lunch") & (tips["sex"]=="Female")].groupby("day").agg({"total_bill":["min","max","sum","mean"],"tip":["min","max","sum","mean"]})

#size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)
#?????????

#total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği totalbill ve tip in toplamını versin
tips["total_bill_sum"] = tips["total_bill"] + tips["tip"]
tips.head()

#total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız
new_tips = tips.sort_values("total_bill_sum",ascending=False)[0:30]
new_tips.shape
new_tips