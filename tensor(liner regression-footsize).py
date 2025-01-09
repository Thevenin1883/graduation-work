import tensorflow as tf

height = [170, 180, 175, 160]
size = [260, 270, 265, 255]

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def loss_func():
    return tf.square(260 - (height * a + b))

opt = tf.keras.optimizers.Adam(learning_rate=0.1)
opt.minimize(loss_func,var_list = [a,b])