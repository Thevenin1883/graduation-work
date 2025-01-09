import tensorflow as tf

height = 170
size = 260
# size = height * a + b 선형 1차 모델
a = tf.Variable(0.1)
b = tf.Variable(0.2)

def loss_func():
    return tf.square(260 - (height * a + b)) #실측값 - 예측값

opt = tf.keras.optimizers.Adam(learning_rate=0.1) # 경사 하강법 

for i in range(300):
    opt.minimize(loss_func,var_list = [a,b])
    print(a.numpy(),b.numpy())