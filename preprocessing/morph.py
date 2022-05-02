#модуль содержит функцию, возвращающую нормальную форму слова

import pymorphy2

def morpholize(word):
    m = pymorphy2.MorphAnalyzer(lang='ru')
    return m.parse(word)[0].normal_form