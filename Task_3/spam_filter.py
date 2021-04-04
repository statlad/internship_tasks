# 3.1 Напишите программу, которая фильтрует спам, поступающий на
# корпоративную почту. Спам – письма, в тексте которых часть букв в
# словах заменена на цифры. На вход подается файл с сообщениями.
# Выход – файл без спам-писем.
# Программа должна работать с файлами, размер которых превышает
# объем оперативной памяти. Файл подается в виде текстового файла.  Одно письмо = одна строка.

# $ python3 main.py data.txt

import sys
import re


def spam_filter(data):
    """
    Spam filter.
    Get .txt file as argv from command line.
    Parameters
    ----------
    data : .txt file

    Returns
    -------
    .txt file
    """
    with open(data, buffering=2000000) as file:
        for line in file:
            op_string = re.sub(r'[^\w\s]', '', line)
            words = [word for word in op_string.split()]
            spam = 0
            for w in words:
                if re.match(r'\d+', w):
                    continue
                if not w.isalpha():
                    spam += 1
                    break

            if spam == 0:
                output.write(line)


if __name__ == "__main__":
    path = sys.argv[1]
    output = open('Output3.txt', 'a+')
    spam_filter(path)
