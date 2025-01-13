import pandas as pd #엑셀 데이터를 다루고 싶을 떄 사용

data = pd.read_csv('gpascore.csv') #데이터 파일 열기

#데이터 전처리 (빈부분을 0으로 처리하거나 평균을 내어서 채움)
data = data.dropna() #빈칸을 없애는 함수

# print(data.isnull().sum())
# print(data['gpa'].min()) .count()
#data = data.fillna(100) 빈칸을 채우는 함수 

yData = data['admit'].values
xData = []
for i, rows in data.iterrows(): #data 프레임을 가로 한줄씩
    xData.append([rows['gre'], rows['gpa'], rows['rank']])

import numpy as np
import tensorflow as tf
model = tf.keras.models.Sequential([tf.keras.layers.Dense(64, activation='tanh'), # 활성함수
                                    tf.keras.layers.Dense(128, activation='tanh'), # 노드의 수 (실험적 파악으로 최적화된 갯수 파악이 필요하다.)
                                    tf.keras.layers.Dense(1, activation='sigmoid')])  # 최종레이어 (본 과제에는 확률 한개를 예측하기 떄문에 1개 필요) 0과 1사이의 값(sigmoid)


model.compile(optimizer='adam' , loss='binary_crossentropy', metrics=['accuracy']) # (binary_crossentropy) loss 함수 0과1 사이의 분류/확률 문제에서 주로쓴다.
model.fit(np.array(xData),np.array(yData),epochs=1000) # fit(x데이터, y데이터, epochs=) x에는 학습 데이터, y에는 실제 정답 데이터, epochs 학습 횟수

# 모델 저장 (HDF5 형식)
model.save('model.h5')
print("Model saved successfully!")

# 모델 로드
loaded_model = tf.keras.models.load_model('model.h5')
print("Model loaded successfully!")

#예측
predictV = model.predict([[750, 3.70, 3],[400, 2.2, 1]])
print(predictV)