# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:37:25 2019

@author: Win7
"""
import kutuphane
from sklearn.feature_selection import SelectKBest, f_classif
#1- Verileri yükle
giris, cikis, kisi_bilgisi =  kutuphane.dosya_oku('data/odev.csv')
olcekli_giris = kutuphane.olceklendir(giris)
dogruluk,f1skor = kutuphane.basari_hesapla(olcekli_giris, cikis, kisi_bilgisi)

#Chi-square 
for k in range(1,30,1):
    ozellikler  = kutuphane.chi2_ozellik_cikar(giris,cikis,k)
    azaltilmis_olcekli_giris = olcekli_giris[:,ozellikler]
    dogruluk,f1skor = kutuphane.basari_hesaplaCV(azaltilmis_olcekli_giris, cikis, kisi_bilgisi,10)
    print("k="+str(k) + " acc="+str(dogruluk))
 
for k in range(1,30,1):
    azaltilmis_olcekli_giris = SelectKBest(f_classif, k=k).fit_transform(X=olcekli_giris,y=cikis)
    dogruluk,f1skor = kutuphane.basari_hesaplaCV(azaltilmis_olcekli_giris, cikis, kisi_bilgisi)
    print("k="+str(k) + " acc="+str(dogruluk))
ozellikler  = kutuphane.chi2_ozellik_cikar(giris,cikis,140)


