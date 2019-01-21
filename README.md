### 手寫辨識

### 環境與套件:
1. Scikit-Learn
2. NUMPY

### Pipeline:
1. 讀入sklearn手寫資料組
2. 將圖片轉為灰階
3. 利用PCA來降維
4. 送入SVM來訓練MODEL

###  細節與參數:
* 訓練資料約為8成=56000筆，測試資料為剩餘兩成=14000筆
* 轉灰階其實就是將原始圖片讀入後將每個圖片中每個PIXEL除以255 讓值介於0~1之間
* 最難的部分是調整PCA的參數，太大要跑非常久尤其是SKLEARN不支援GPU加速 太小根本學不到東西
* 有試過把PCA的N_COMPONENT參數調為1，明顯沒用，完全沒學習到
  ![FAIL](https://github.com/jt851113/ML2018_410421233/blob/master/photo/FAIL.png)
* 訓練過程

  ![Process](https://github.com/jt851113/ML2018_410421233/blob/master/photo/PROCESS.JPG)

* 最後結果
  ![FINAL](https://github.com/jt851113/ML2018_410421233/blob/master/photo/final.JPG)
  算不上好而且還有很大進步空間，大概要透過SVM的設置來改善
  
### 雜談:
這次不算太難，主要是調整pca降維那邊很難調整，若狀況允許，當然直接扁平化是最好，這樣能夠保留完整的特徵值
但在跟速度的妥協下，還是讓他減少維度，因為之前有練習過用keras的手寫辨識，跟那次不同，那次是用神經網路來實作
跟這次用分類器做的感覺不太一樣，當然這也是個很好的學習機會，讓我能夠更加熟悉python的套件以及複習之前上課所學到的
