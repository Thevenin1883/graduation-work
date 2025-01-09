import tensorflow as tf

# 텐서를 이용한 행렬 표현 방법
tensor1 = tf.constant([3,4,5])
tensor2 = tf.constant([6,7,8])
tensor3 = tf.constant( [[1,2],
                        [3,4]])
tensor4 = tf. constant([[1,2],
                        [3,4]])

# 텐서 연산
# tf.add(), tf.subtract(), tf.divide(), tf.multiply()
print(tf.subtract(tensor1, tensor2))

#텐서 행렬 연산 tf.matmul

print(tf.matmul(tensor3, tensor4))

#0 이 가득한 텐서 생성 
tensor5 = tf.zeros(10) 
print(tensor5)

tensor6 = tf.zeros([2,2])
print(tensor6)

tensor7 = tf.zeros([2,2,3])
print(tensor7)

#tensor의 shape
print(tensor1.shape)
print(tensor7.shape)


#tensor의 자료형
#dtype = (정보) <= int32, float32.. 대부분 실수 타입을 사용한다.
#type의 변환
# tensor = tf.constant([3,4,5], tf.float) <<< 다음과 같이 or
# tf.cast() 함수 사용

w = tf.Variable(1.0) # 변수 즉 Weight값 변경이 되는 값은 Variable 타입으로 선언
print(w.numpy()) # 변수에 저장되어 있는 값 가져오는 법
w.assign(2) # 변수에 값 재할당 하는 법
print(w.numpy())

#Variable도 tensor와 같은 행렬로 표현가능