# Данные параметры доступны для изменений.
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
load_model = False

image_path = "internship_data/" # Название папки с данными

# Структура папок

Файл для запуска тренировки
root/../ProjectFolder/train.py

Файл для процессинга новых изображений
root/../ProjectFolder/process.py

Необходимая структура папок с данными для запуска обучения
root/../ProjectFolder/internship_data/Female/ .jpeg
......./ProjectFolder/internship_data/Male/ .jpeg

# Запуск обучения
 - Тренировочный файл нужно положить в папку ProjectFolder
$ cd ../internship_tasks/Task 2
$ python3 train.py

# Скрипт для классификации новых изображений

$ python3 process.py folder/to/process

# [folder/to/process] -> путь к папке с файлами для классификации
Результатом будет process_results.json с ответами вида {'image1.jpeg':'female'}
