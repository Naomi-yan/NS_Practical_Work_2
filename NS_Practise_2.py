# Импорт необходимых библиотек
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow import keras
from tensorflow.keras import layers, models, utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping

# Загрузка и подготовка данных MNIST
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

# Нормализация данных (приведение к диапазону [0, 1])
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Преобразование формата изображений (добавление размерности канала)
x_train_full = x_train_full.reshape((x_train_full.shape[0], 28*28))
x_test = x_test.reshape((x_test.shape[0], 28*28))

# Преобразование меток в one-hot encoding
y_train_full = utils.to_categorical(y_train_full, 10)
y_test = utils.to_categorical(y_test, 10)

# Разделение обучающей выборки на обучающую и валидационную (10 000 образцов)
indices = np.random.permutation(len(x_train_full))
x_val = x_train_full[indices[:10000]]
y_val = y_train_full[indices[:10000]]
x_train = x_train_full[indices[10000:]]
y_train = y_train_full[indices[10000:]]

print(f"Размеры выборок:")
print(f"Обучающая: {x_train.shape}, {y_train.shape}")
print(f"Валидационная: {x_val.shape}, {y_val.shape}")
print(f"Тестовая: {x_test.shape}, {y_test.shape}")

# Создание архитектуры нейронной сети
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(28*28,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Вывод структуры сети
model.summary()

# Компиляция модели
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Обучение с контролем переобучения (ранняя остановка)
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# Оценка точности на тестовой выборке
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nТочность на тестовой выборке: {test_acc:.4f}")

# Визуализация процесса обучения
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Точность на обучении')
plt.plot(history.history['val_accuracy'], label='Точность на валидации')
plt.title('Точность модели')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Потери на обучении')
plt.plot(history.history['val_loss'], label='Потери на валидации')
plt.title('Потери модели')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Дополнительное задание 9: Загрузка и классификация собственного изображения
def load_and_predict_digit(image_path):
    # Загрузка изображения
    img = Image.open(image_path).convert('L')  # Конвертация в градации серого
    img = img.resize((28, 28))  # Изменение размера до 28x28
    
    # Преобразование в массив
    img_array = np.array(img)
    
    # Инвертирование цветов (если фон белый, а цифра черная)
    img_array = 255 - img_array
    
    # Нормализация
    img_array = img_array.astype('float32') / 255.0
    
    # Изменение формы для модели
    img_array = img_array.reshape(1, 28*28)
    
    # Предсказание
    predictions = model.predict(img_array, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # Визуализация
    plt.figure(figsize=(6, 3))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_array.reshape(28, 28), cmap='gray')
    plt.title(f'Загруженное изображение')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(10), predictions[0])
    plt.title(f'Предсказание: {predicted_digit} ({confidence:.2%})')
    plt.xlabel('Цифра')
    plt.ylabel('Вероятность')
    plt.xticks(range(10))
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return predicted_digit, confidence, predictions[0]

# Пример использования (замените путь на путь к вашему изображению)
try:
    digit, confidence, all_predictions = load_and_predict_digit('my_digit.png')
    print(f"\nРезультат классификации собственной цифры:")
    print(f"Предсказанная цифра: {digit}")
    print(f"Уверенность: {confidence:.2%}")
    print(f"Все вероятности: {all_predictions}")
except FileNotFoundError:
    print("\nФайл 'my_digit.png' не найден. Создайте изображение цифры 28x28 пикселей.")
    
# Вывод нескольких неверно классифицированных изображений
print("\nАнализ ошибок классификации...")
predictions = model.predict(x_test, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Поиск индексов неверных предсказаний
incorrect_indices = np.where(predicted_labels != true_labels)[0]

print(f"Всего ошибок: {len(incorrect_indices)} из {len(x_test)} ({len(incorrect_indices)/len(x_test)*100:.2f}%)")

# Вывод первых 5 ошибок
plt.figure(figsize=(15, 6))
for i, idx in enumerate(incorrect_indices[:5]):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f'Истинная: {true_labels[idx]}\nПредсказанная: {predicted_labels[idx]}')
    plt.axis('off')
plt.suptitle('Первые 5 неверно классифицированных изображений')
plt.tight_layout()
plt.show()

# Детальный анализ для каждой ошибки
print("\nДетальный анализ первых 5 ошибок:")
for i, idx in enumerate(incorrect_indices[:5]):
    print(f"\nОшибка {i+1}:")
    print(f"  Индекс: {idx}")
    print(f"  Истинная цифра: {true_labels[idx]}")
    print(f"  Предсказанная цифра: {predicted_labels[idx]}")
    print(f"  Вероятности: {predictions[idx]}")