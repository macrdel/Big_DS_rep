import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Flatten, Reshape, Dropout
from keras.models import Model
from sys import exit
import time

img_rows = img_cols = 28
mnist = not True
flatten = not True
reshape = True
pathToData = 'mnist/' if mnist else 'emnist/'
num_classes = 10 if mnist else 26
epochs = 35  # if mnist else 40
fn_model = pathToData + 'lk3.h5'
#
pathToHistory = 'mnist/' if mnist else 'emnist/'
suff = '.txt'
# Имена файлов, в которые сохраняется история обучения
fn_loss = pathToHistory + 'loss_' + suff
fn_acc = pathToHistory + 'acc_' + suff
fn_val_loss = pathToHistory + 'val_loss_' + suff
fn_val_acc = pathToHistory + 'val_acc_' + suff


def one_plot(n, y_lb, loss_acc, val_loss_acc):
    plt.subplot(1, 2, n)
    if n == 1:
        lb, lb2 = 'loss', 'val_loss'
        yMin = 0
        yMax = 1.05 * max(max(loss_acc), max(val_loss_acc))
    else:
        lb, lb2 = 'acc', 'val_acc'
        yMin = min(min(loss_acc), min(val_loss_acc))
        yMax = 1.0
    plt.plot(loss_acc, color='r', label=lb, linestyle='--')
    plt.plot(val_loss_acc, color='g', label=lb2)
    plt.ylabel(y_lb)
    plt.xlabel('Эпоха')
    plt.ylim([0.95 * yMin, yMax])
    plt.legend()


def loadBinData(pathToData, img_rows, img_cols):
    print('Загрузка данных из двоичных файлов...')
    with open(pathToData + 'imagesTrain.bin', 'rb') as read_binary:
        x_train = np.fromfile(read_binary, dtype=np.uint8)
    with open(pathToData + 'labelsTrain.bin', 'rb') as read_binary:
        y_train = np.fromfile(read_binary, dtype=np.uint8)
    with open(pathToData + 'imagesTest.bin', 'rb') as read_binary:
        x_test = np.fromfile(read_binary, dtype=np.uint8)
    with open(pathToData + 'labelsTest.bin', 'rb') as read_binary:
        y_test = np.fromfile(read_binary, dtype=np.uint8)
        # Преобразование целочисленных данных в float32 и нормализация; данные лежат в диапазоне [0.0, 1.0]
    x_train = np.array(x_train, dtype='float32') / 255
    x_test = np.array(x_test, dtype='float32') / 255
    if mnist:
        x_train = x_train.reshape(-1, img_rows, img_cols)
        x_test = x_test.reshape(-1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(-1, img_rows, img_cols, 1).transpose(0, 2, 1, 3)
        x_test = x_test.reshape(-1, img_rows, img_cols, 1).transpose(0, 2, 1, 3)
    # Преобразование в категориальное представление: метки - числа из диапазона [0, 9] в двоичный вектор размера num_classes
    # Так, в случае MNIST метка 5 (соответствует классу 6) будет преобразована в вектор [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
    # print(y_train[0]) # (MNIST) Напечатает: 5
    print('Преобразуем массивы меток в категориальное представление')
    if not mnist:
        y_train -= 1
        y_test -= 1
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    # print(y_train[0]) # (MNIST) Напечатает: [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = loadBinData(pathToData, img_rows, img_cols)


input_shape = (img_rows, img_cols)
#
# Создание модели нейронной сети
inp = Input(shape=input_shape)  # Входной слой
x = inp
if flatten:
    x = Flatten()(x)  # Преобразование 2D в 1D
elif reshape:
    x = Reshape((-1, ))(x)
x = Dense(units=256, activation='relu')(x)
x = Dropout(0.15)(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=inp, outputs=output)
model.summary()


model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])



start = time.time()
history = model.fit(x_train, y_train, batch_size=128, epochs=epochs,
                    verbose=2, validation_data=(x_test, y_test))
print('Время вычислений:', time.time() - start)
print('Модель сохранена в файле', fn_model)
model.save(fn_model)


history = history.history
## for itm in history.items(): print(itm)
with open(fn_loss, 'w') as output:
    for val in history['loss']: output.write(str(val) + '\n')
with open(fn_acc, 'w') as output:
    for val in history['accuracy']: output.write(str(val) + '\n')
with open(fn_val_loss, 'w') as output:
    for val in history['val_loss']: output.write(str(val) + '\n')
with open(fn_val_acc, 'w') as output:
    for val in history['val_accuracy']: output.write(str(val) + '\n')


# Вывод графиков обучения
plt.figure(figsize=(9, 4))
plt.subplots_adjust(wspace=0.5)
one_plot(1, 'Потери', history['loss'], history['val_loss'])
one_plot(2, 'Точность', history['accuracy'], history['val_accuracy'])
plt.suptitle('Потери и точность')
plt.show()


# pred = model.predict(x_train)
# predicted_classes = np.array([np.argmax(m) for m in pred])
# if not mnist:
#     import string
#     letters = list(string.ascii_uppercase)
#     predicted_classes = [letters[m] for m in predicted_classes]
# l = len(predicted_classes)
#
#
# f = plt.figure(figsize=(12, 5))
# for i in range(15):
#     j = np.random.randint(0, high=l, dtype=int)
#     sp = f.add_subplot(1, 15, i+1)
#     sp.axis('Off')
#     sp.set_title(predicted_classes[j])
#     plt.imshow(x_train[j], cmap='gray')
# plt.show()


pred = model.predict(x_test)
predicted_classes = np.array([np.argmax(m) for m in pred])
if not mnist:
    import string
    letters = list(string.ascii_uppercase)
    predicted_classes = [letters[m] for m in predicted_classes]
l = len(predicted_classes)


f = plt.figure(figsize=(12, 5))
for i in range(15):
    j = np.random.randint(0, high=l, dtype=int)
    sp = f.add_subplot(1, 15, i+1)
    sp.axis('Off')
    sp.set_title(predicted_classes[j])
    plt.imshow(x_test[j], cmap='gray')
plt.show()

# for emnist
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
mask = []
for i in range(len(pred)):
    mask.append(pred[i] == np.argmax(y_test[i]))
x_false = []
for i in range(len(x_test)):
    if not mask[i]:
        x_false.append(x_test[i])
preds_incorrect = []
for i in range(len(x_test)):
    if not mask[i]:
        preds_incorrect.append(pred[i])
y_test_true = []
for i in range(len(x_test)):
    if not mask[i]:
        y_test_true.append(y_test[i])
# вывод некорректных данных
l = len(y_test_true)
f = plt.figure(figsize=(12,5))
for i in range(10):
    j = np.random.randint(0, high=l, dtype=int)
    sp = f.add_subplot(1,10,i+1)
    sp.axis('Off')
    sp.set_title("{} | {}".format(letters[preds_incorrect[j]], letters[np.argmax(y_test_true[j])]))
    plt.imshow(x_false[j], 'gray')
plt.show()


# подсчёт точности по классам
l = len(y_test)
i = 1
k = 0
acc = []
tmp_acc = 0
for j in range(l):
    if mask[j]:
        tmp_acc += 1
    if j == (799 * i) + k:
        i += 1
        k += 1
        acc.append((tmp_acc * 100) / 800)
        tmp_acc = 0
print(acc)