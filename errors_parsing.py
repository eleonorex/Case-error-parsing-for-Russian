import spacy
from spacy.matcher import DependencyMatcher
import pickle
from internship2022.Sentence import Sentence
from natasha import Doc, Segmenter, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, MorphVocab
import pymorphy2

nlp = spacy.load("ru_core_news_sm")

pattern1_dict = {}
prepositions = {}


def find_pattern1(text):
    """находит сочетания ГЛАГОЛ + ПРЯМОЕ ДОПОЛНЕНИЕ"""

    verb_obj = [
        {
            "RIGHT_ID": "глагол",
            "RIGHT_ATTRS": {"POS": "VERB"}
        },
        {
            "LEFT_ID": "глагол",
            "REL_OP": ">",
            "RIGHT_ID": "дополнение",
            "RIGHT_ATTRS": {"DEP": "obj"},
        }
    ]

    doc = nlp(text)
    dep_matcher = DependencyMatcher(nlp.vocab)
    dep_matcher.add("verb_compl", [verb_obj])
    matches = dep_matcher(doc)

    # создаем словарь
    for match in matches[0:]:
        match_id, token_ids = match
        for i in range(len(token_ids)):
            dict_morph = doc[token_ids[1]].morph.to_dict()
            try:
                pattern1_dict[doc[token_ids[0]].lemma_] = {
                    '_': [doc[token_ids[1]].lemma_, dict_morph["Case"]]}

                prepositions[doc[token_ids[0]].lemma_] = "_"
            except KeyError:
                pass

    return pattern1_dict


pattern2_dict = {}


def find_pattern2(text):
    """находит только те сочетания ГЛАГОЛ+ПРЕДЛОГ+ДОП-Е, где все токены идут друг за другом,
     иначе мэтчит слишком много сочетаний с ошибками"""

    verb_prep_obj = [
        {
            "RIGHT_ID": "глагол",
            "RIGHT_ATTRS": {"DEP": {'IN': ["ROOT", "xcomp"]}}
        },
        {
            "LEFT_ID": "глагол",
            "REL_OP": ".",
            "RIGHT_ID": "предлог",
            "RIGHT_ATTRS": {"DEP": "case"},
        },
        {
            "LEFT_ID": "предлог",
            "REL_OP": ".",
            "RIGHT_ID": "дополнение",
            "RIGHT_ATTRS": {"DEP": "obl"},
        }
    ]

    doc = nlp(text)
    dep_matcher = DependencyMatcher(nlp.vocab)
    dep_matcher.add("verb_prep_compl", [verb_prep_obj])
    matches = dep_matcher(doc)

    # печатаем все результаты
    # for match in matches[0:]:
    #     match_id, token_ids = match
    #     for i in range(len(token_ids)):
    #         print(verb_prep_obj[i]["RIGHT_ID"] + ":", doc[token_ids[i]].text)

    # создаем словарь
    for match in matches[0:]:
        match_id, token_ids = match
        for i in range(len(token_ids)):
            dict_morph = doc[token_ids[2]].morph.to_dict()
            pattern2_dict[doc[token_ids[0]].lemma_] = {
                doc[token_ids[1]].lemma_: [doc[token_ids[2]].lemma_, dict_morph["Case"]]}

            prepositions[doc[token_ids[0]].lemma_] = doc[token_ids[1]].lemma_

    return pattern2_dict


with open('test_text.txt', 'r') as kryzhovnik:
    text1 = kryzhovnik.read()

# добавляем по одной ошибке в каждое предложение
text_errors = {}
mistakes = []
var = []
for phrase in text1.split('.'):
    sentence = Sentence(phrase)
    sentence.text = phrase
    var = sentence.change_word
    print(var)
    if sentence.replaced is not None:
        mistakes.append(sentence.replaced)

    text_errors[var[0]] = var[1]

new_text = '.'.join(text_errors.keys())
print(mistakes)

with open('comb_verb_noun.pickle', 'rb') as f:
    correction_data = pickle.load(f)

# списки со статистикой по словосочетаниям
sure = []
possible = []
not_stated = []

# проверяем прямые дополнения
direct = find_pattern1(new_text)

for verb in direct.keys():
    verb_depends = direct[verb]
    dictionary = verb_depends[prepositions[verb]]
    case = dictionary[1]
    obj = dictionary[0]

    try:
        dependencies = correction_data[verb]
        relation = [x for x in direct[verb].keys()]
        dict2 = dependencies[relation[0]]
        if case in dict2.keys():
            # если прямые дополнения есть в данных этого глагола
            if obj in dict2[case].keys():
                # если такое дополнение есть в данных этого глагола + падежа
                sure.append([verb, case, obj])
            else:
                # в данных нет такого дополнения, но прямые доп-я допускаются
                possible.append([verb, case, obj])
        else:
            # в данных нет прямых дополнений
            not_stated.append([verb, case, obj])
    except KeyError:
        # в данных нет употреблений этого глагола с таким предлогом
        not_stated.append([verb, case, obj])

# проверяем дополнения с предлогом
indirect = find_pattern2(new_text)

for verb in indirect.keys():
    verb_depends = indirect[verb]
    dictionary = verb_depends[prepositions[verb]]
    case = dictionary[1]
    obj = dictionary[0]
    try:
        dependencies = correction_data[verb]
        relation = [x for x in indirect[verb].keys()]

        try:
            dict2 = dependencies[relation[0]]
            if case in dict2.keys():
                # если такой падеж есть в данных этого глагола
                if obj in dict2[case].keys():
                    # если такое дополнение есть в данных этого глагола + падежа
                    sure.append([verb, case, obj])
                else:
                    # в данных есть такой падеж, но нет именно такого дополнения
                    possible.append([verb, case, obj])
            else:
                # в данных нет такого падежа в сочетании с этим глаголом
                not_stated.append([verb, case, obj])
        except KeyError:
            # в данных нет употреблений этого глагола с таким предлогом
            not_stated.append([verb, case, obj])

    # если мы по ошибке приняли что-то за глагол или такого глагола нет в списке
    except KeyError:
        pass

# --------------------------------------------- ЧАСТЬ С ПОИСКОМ СЛОВОСОЧЕТАНИЙ
morph = pymorphy2.MorphAnalyzer()
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
doc = Doc(text1)
doc.segment(segmenter)
doc.tag_morph(morph_tagger)
doc.parse_syntax(syntax_parser)

list_1 = []
dict_example = {"WORD": None, "PREP": None, "CASE": None, "WORD1": None}
'''
Если у токена тип отношения nmod, appos, case, conj или det:
    Если это существительное NOUN или PROPN:
        добавить в массив словарь вида [слово, "_", падеж зависимого слова, зависимое слово]
    Если это предлог ADP:
        найти слово, которому принадлежит этот предлог, через head_id
        добавить в массив словарь вида [слово, предлог, падеж зависимого слова, зависимое слово]

'''

for token in doc.tokens:
    if token.rel in ["nmod", "appos", "conj", "case", "det"] and token.pos in ["NOUN", "PROPN", "ADP"]:
        word, prep, case, word2 = "", "", "", ""
        if token.pos == "ADP":
            prep = token.text.upper()
            # ищем кому принадлежит предлог
            for token_n in doc.tokens:
                if token_n.id == token.head_id:  # если предлог ссылается на это слово
                    token_n.lemmatize(morph_vocab)
                    word2 = token_n.lemma.lower()
                    if "Case" in token_n.feats:
                        case = token_n.feats["Case"].title()
                    # теперь ищем кому принадлежит это зависимое слово
                    for token_A in doc.tokens:
                        if token_A.id == token_n.head_id and token_A.pos in ["NOUN", "PROPN"]:
                            token_A.lemmatize(morph_vocab)
                            word = token_A.lemma.lower()
                        pass
        else:  # if token.pos in ["NOUN", "PROPN"]
            prep = "_"
            token.lemmatize(morph_vocab)
            word2 = token.lemma.lower()
            case = token.feats["Case"].title()
            for token_n in doc.tokens:
                if token_n.id == token.head_id:
                    token_n.lemmatize(morph_vocab)
                    word = token_n.lemma.lower()

        if word != "":
            dict_to_add = {"WORD": word, "PREP": prep, "CASE": case, "WORD2": word2}
            list_1.append(dict_to_add)

with open("comb_noun_noun.pickle", "rb") as file:
    data = pickle.load(file)

for X in list_1:

    if X["WORD"] in data:

        if X["PREP"] in data[X["WORD"]]:

            if X["CASE"] in data[X["WORD"]][X["PREP"]]:

                if X["WORD2"] in data[X["WORD"]][X["PREP"]][X["CASE"]]:

                    sure.append([X["WORD"], X["PREP"], X["CASE"], X["WORD2"]])

                else:
                    possible.append([X["WORD"], X["PREP"], X["CASE"], X["WORD2"]])
            else:
                not_stated.append([X["WORD"], X["PREP"], X["CASE"], X["WORD2"]])
        else:
            not_stated.append([X["WORD"], X["PREP"], X["CASE"], X["WORD2"]])
    else:
        not_stated.append([X["WORD"], X["PREP"], X["CASE"], X["WORD2"]])

list_2 = []
dict_example = {"VERB": None, "PREP": None, "CASE": None, "WORD1": None}
for token in doc.tokens:  # перебираем токены предложения
    verb, prep, case, word2 = "", "", "", ""
    if token.pos == "VERB":  # если нашли глагол
        token.lemmatize(morph_vocab)
        verb = token.lemma
        for token_a in doc.tokens:  # перебираем снова в поиске существительных, ссылающихся на наш глагол
            if token_a.head_id == token.id and token_a.rel in ["obj", "obl"]:
                token_a.lemmatize(morph_vocab)
                word2 = token_a.lemma
                if 'Case' in token_a.feats:
                    case = token_a.feats["Case"]
                for token_b in doc.tokens:  # перебираем снова в поиске предлогов, ссылающихся на наше существительное
                    if token_b.pos == "ADP" and token_b.head_id == token_a.id:
                        prep = token_b.text
                        break
                    else:
                        prep = "_"

        if word2 != "":
            dict_to_add = {"VERB": verb, "PREP": prep, "CASE": case, "WORD2": word2}
            list_2.append(dict_to_add)

sure, possible, not_stated = [], [], []
for X in list_2:

    if X["VERB"] in correction_data:

        if X["PREP"] in correction_data[X["VERB"]]:

            if X["CASE"] in correction_data[X["VERB"]][X["PREP"]]:

                if X["WORD2"] in correction_data[X["VERB"]][X["PREP"]][X["CASE"]]:
                    sure.append([X["VERB"], X["PREP"], X["CASE"], X["WORD2"]])

                else:
                    possible.append([X["VERB"], X["PREP"], X["CASE"], X["WORD2"]])
            else:
                not_stated.append([X["VERB"], X["PREP"], X["CASE"], X["WORD2"]])
        else:
            not_stated.append([X["VERB"], X["PREP"], X["CASE"], X["WORD2"]])
    else:
        not_stated.append([X["VERB"], X["PREP"], X["CASE"], X["WORD2"]])

found_mistakes = []
missed_mistakes = []

for mistake in mistakes:
    doc = nlp(mistake)
    for token in doc:
        for group in not_stated:
            if token.lemma_ in group:
                found_mistakes.append(group)
        for group in sure:
            if token.lemma_ in group:
                missed_mistakes.append(group)
        for group in possible:
            if token.lemma_ in group:
                missed_mistakes.append(group)

with open("RESULT.txt", "w") as result:
    result.write(f'Всего словосочетаний:{len(sure) + len(possible) + len(not_stated)} \n')
    result.write(f'{len(sure)} примеров есть в словаре: {sure} \n')
    result.write(f'{len(possible)} допустимых употреблений: {possible} \n')
    result.write(f'Для {len(not_stated)} данных нет, вероятно, ошибка: {not_stated} \n \n')
    result.write(f'Сгенерированные ошибки: {len(mistakes)} \n')
    result.write(f'Найденные ошибки: {len(found_mistakes)} \n {found_mistakes} \n')
    result.write(f'Пропущенные ошибки: {len(missed_mistakes)} \n {missed_mistakes} \n')
    result.write(f'Эффективность поиска ошибок: {100 * (float(len(found_mistakes)) / float(len(mistakes)))} % \n')
