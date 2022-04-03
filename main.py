import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist  # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

#загрузка обучающего тестового множества
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# mnist.load_data - загрузка изображений из библиотеки
# x_train - изображение с цифрой обучающей выборки
# y_train - значение изображения на картинки из x_train обучающей выборки
# x_test - изображение с цифрой тестовой выборки
# y_test - значение изображения на картинки из x_train тестовой выборки

# стандартизация входных данных
# нормализация входных данных
x_train = x_train / 255  # вещественные числа от 0 до 1
x_test = x_test / 255  # вещественные числа от 0 до 1


# преобразование выходных значений в векторы, (от 0 до 9) единица на месте нужного числа
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)



# формирование модели нейронной сети
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),  # преобразование изображения для нейронной сети 28х28 пикселей (1 - 1 байт, один пиксель в градации серого (от 0 до 255))
    Dense(128, activation='relu'),  # связь входных значений и скрытого слоя
    Dense(10, activation='softmax')  # связь скрытого слоя и выходного слоя
])

print(model.summary())      # вывод структуры НС в консоль


# компиляция НС с оптимизацией по Adam
model.compile(optimizer='adam',
             loss='categorical_crossentropy',  # критерий категориальная кросс-энтропия
             metrics=['accuracy'])  # отслеживание % правильный решений



# запуск процесса обучения
model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
# x_train - изображение с цифрой обучающей выборки
# y_train_cat - преобразование выходных значений в векторы, (от 0 до 9) единица на месте нужного числа
# batch_size - после 32 изображений корректируем весовые коэфф.
# epochs - количество эпох (качество обучения НС, количество итераций)
# validation_split - разбиение выборки на обучающую и проверочную


# работа с тестовой выборкой
model.evaluate(x_test, y_test_cat)




n = 9  # тут передаем изображение
x = np.expand_dims(x_test[n], axis=0)  # создаем трехмерный тензор
res = model.predict(x)
print( res )
print( np.argmax(res) )  # через argmax выбираем индекс максимального значения

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()


