# При обучении выводить потери и точность и генератора, и дискриминатора
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sys import exit

pathToData = 'C:\\_B\\AM\\CG\\mnist\\'
pathToHistory = ''
num_classes = 10
optimizer = Adam(0.0002, 0.5)
loss = 'binary_crossentropy'
loss_g = 'binary_crossentropy'  # 'mse', 'poisson', 'binary_crossentropy'
# latent_dim - размер массива, подаваемого на вход генератора
latent_dim = 100
epochs = 30001  # Число эпох обучения на пакете
batch_size = 32  # Размер пакета обучения (число генерируемых изображений)
sample_interval = 3000  # Интервал между сохранением сгенерированных изображений в файл
show_img = False
show_sum = not False
file_gen = pathToHistory + 'generator_model_%03d.h5' % epochs
print('epochs =', epochs, '\ batch_size =', batch_size)


def load_data(pathToData, num_classes, show_img):
    print('Загрузка данных')
    with open(pathToData + 'imagesTrain.bin', 'rb') as read_binary:
        x_trn = np.fromfile(read_binary, dtype=np.uint8)
    x_trn = x_trn.reshape(-1, 28 * 28)
    x_trn = x_trn / 255
    if show_img:
        r = c = 5
        with open(pathToData + 'labelsTrain.bin', 'rb') as read_binary:
            y_trn = np.fromfile(read_binary, dtype=np.uint8)
        for i, j in enumerate(np.random.randint(0, len(x_trn), r * c)):
            plt.subplot(r, c, i + 1)
            plt.imshow(x_trn[j].reshape(28, 28), cmap='gray')
            plt.title(y_trn[j])
            plt.axis('off')
        plt.subplots_adjust(hspace=0.5)  # wspace
        plt.show()
        exit()
    # Приводим к диапазону [-1, 1]; activation = 'sigmoid' & activation = 'tanh'
    x_trn = 2.0 * x_trn - 1.0
    return x_trn


def m_inp(x, b_s, l_dim):  # Вход дискриминатора и генератора
    idx = np.random.randint(0, len(x), b_s)
    d_n = x[idx]
    g_n = np.random.normal(0, 1, (b_s, l_dim))
    return d_n, g_n


def one_dense(x, units, m, a):
    x = Dense(units)(x)
    x = LeakyReLU(alpha=a)(x)
    x = BatchNormalization(momentum=m)(x)
    return x


def one_dense2(x, units, a):
    x = Dense(units)(x)
    x = LeakyReLU(alpha=a)(x)
    return x


def build_generator(latent_dim):
    inp = Input(latent_dim)
    x = inp
    x = one_dense(x, 256, 0.8, 0.2)
    x = one_dense(x, 512, 0.8, 0.2)
    x = one_dense(x, 1024, 0.8, 0.2)
    out = Dense(784, activation='tanh')(x)
    generator = Model(inp, out)
    if show_sum:
        generator.summary()
    return generator


def build_discriminator(loss, optimizer):
    inp = Input(784)
    x = inp
    x = one_dense2(x, 512, 0.2)
    x = one_dense2(x, 256, 0.2)
    out = Dense(1, activation='sigmoid')(x)
    discriminator = Model(inp, out)
    if show_sum:
        discriminator.summary()
    discriminator.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return discriminator


def train(discriminator, generator, combined, epochs, b_s,
          sample_interval, l_dim, pathToHistory, x_trn):
    # Метки истинных и ложных изображений
    valid = np.ones(b_s)
    d_loss, d_acc, g_loss, g_acc = [], [], [], []
    for epoch in range(epochs):
        # Обучаем дискриминатор
        d_n, g_n = m_inp(x_trn, b_s, l_dim)  # Вход дискриминатора (d_n) и генератора (g_n)
        gen_imgs = generator.predict(g_n)  # Генерируем batch_size изображений
        # Обучаем дискриминатор, подавая ему сначала настоящие, а затем поддельные изображения
        d_hist_real = discriminator.train_on_batch(d_n, valid)
        d_hist_fake = discriminator.train_on_batch(gen_imgs, valid - 1)
        # Усредняем результаты и получаем средние потери и точность
        d_ls, d_a = 0.5 * np.add(d_hist_real, d_hist_fake)
        # Обучение обобщенной модели. Реально обучается только генератор
        _, g_n = m_inp(x_trn, b_s, l_dim)
        g_ls, g_a = combined.train_on_batch(g_n, valid)
        if epoch % 100 == 0:
            d_loss.append(d_ls)
            d_acc.append(d_a)
            g_loss.append(g_ls)
            g_acc.append(g_a)
        # Потери и точность дискриминатора и потери генератора
        if epoch % (sample_interval / 10) == 0:
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, acc.: %.2f%%]"
                  % (epoch, d_ls, 100 * d_a, g_ls, 100 * g_a))
        # Генерируем и сохраняем рисунок с 25-ю изображениями
        if epoch % sample_interval == 0:
            save_sample_images(latent_dim, generator, epoch, x_trn)
    # Сохраняем обученный генератор в файл
    generator.compile()
    generator.save(file_gen)
    print('Модель генератора сохранена в файл', file_gen)
    # Вывод историй обучения в файлы
    fn_d_loss, fn_d_acc, fn_g_loss = 'd_loss.txt', 'd_acc.txt', 'g_loss.txt'
    print('Истории сохранены в файлы:\n' + fn_d_loss + '\n' + fn_d_acc + '\n' + fn_g_loss)
    with open(pathToHistory + fn_d_loss, 'w') as output:
        for val in d_loss:
            output.write(str(val) + '\n')
    with open(pathToHistory + fn_d_acc, 'w') as output:
        for val in d_acc:
            output.write(str(val) + '\n')
    with open(pathToHistory + fn_g_loss, 'w') as output:
        for val in g_loss:
            output.write(str(val) + '\n')
    # Вывод графиков историй обучения
    print('Вывод графиков историй обучения')
    yMax = max(g_loss)
    cnt = len(g_loss)
    rng = np.arange(cnt)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(rng, d_loss, marker='o', c='blue', edgecolor='black')
    ax.scatter(rng, g_loss, marker='x', c='red')
    ax.set_title('Потери генератора (x) и дискриминатора (o)')
    ax.set_ylabel('Потери')
    ax.set_xlabel('Эпоха / 100')
    ax.set_xlim([-0.5, cnt])
    ax.set_ylim([0, 1.1 * yMax])
    fig.show()


def save_sample_images(l_dim, generator, epoch, x_trn):
    r, c = 5, 5  # Выводим и сохраняем 25 изображений
    _, g_n = m_inp(x_trn, r * c, l_dim)
    gen_imgs = generator.predict(g_n)
    # Возвращаемся к диапазону [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt].reshape(28, 28), cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    # Сохраняем изображения
    fig.savefig(pathToHistory + '%d.png' % epoch)
    plt.close()


generator = build_generator(latent_dim)  # Генератор
discriminator = build_discriminator(loss, optimizer)  # Дискриминатор (c компиляцией)
# Обобщенная модель
inp = Input(latent_dim)
x = generator(inp)
discriminator.trainable = False
out = discriminator(x)
combined = Model(inp, out)
combined.compile(loss=loss_g, optimizer=optimizer, metrics=['accuracy'])
if show_sum:
    combined.summary()
x_trn = load_data(pathToData, num_classes, show_img)
print('Обучение. Интервал между выводом изображений ', sample_interval)
train(discriminator, generator, combined, epochs,
      batch_size, sample_interval, latent_dim, pathToHistory, x_trn)