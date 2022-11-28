from natasha import Doc, Segmenter, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, MorphVocab
import pymorphy2
import random


class Sentence:
    morph = pymorphy2.MorphAnalyzer()
    segmenter = Segmenter()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    syntax_parser = NewsSyntaxParser(emb)
    morph_vocab = MorphVocab()
    list_of_cases = ["nomn", "gent", "datv", "accs", "ablt", "loct"]

    def __init__(self, text):
        self.text = text  # нетронутый текст
        self.doc = Doc(text)  # текст, который мы будем обрабатывать
        self.token_array = []  # будущий массив токенов
        self.newtext = text  # текст для замены в нём слова
        self.mistakeID = None  # позиция изменённого токена
        self.replaced = None

    def preprocess(self):  # препроцессинг вынесен в отдельную ф-цию для лучшей читаемости кода
        self.doc.segment(self.segmenter)
        self.doc.tag_morph(self.morph_tagger)
        self.doc.parse_syntax(self.syntax_parser)
        self.token_array = self.doc.tokens  # каждый токен хранит в себе текст + его грамматические харак-ки

    @property
    def change_word(self):
        self.preprocess()
        a = [_.text for _ in self.token_array]  # вытаскиваем из каждого токена непосредственно строку с текстом
        tokens_to_inflect = []
        dependencies_dict = {}

        '''добавляем слова из выбранной части речи в кандидаты на склонение'''
        for token in self.token_array:
            if (token.pos == "NOUN" or token.pos == "PROPN") and (token.rel == "obl" or token.rel == "obj"):
                tokens_to_inflect.append(token)

        '''случайным образом выбираем слово из списка подходящих'''
        try:
            word = random.choice(tokens_to_inflect).text
            word_ID = a.index(word)


            '''ОБРАБОТКА'''
            parsed_word = self.morph.parse(word)[0]
            # case = parsed_word.tag.case
            try:
                declensions = [parsed_word.inflect({case}).word for case in self.list_of_cases]  # склоняем слово
                for x in declensions:
                    if x == word or x == parsed_word.inflect({"plur"}):
                        declensions.remove(x)  # убираем склонения, совпадавшие по форме слова с оригиналом

                newWord = random.choice(declensions)
                self.replaced = newWord
                self.newtext = self.newtext.replace(word, newWord)
                self.mistakeID = word_ID
                return [self.newtext, self.mistakeID]

            except IndexError:
                self.mistakeID = '-'
                return [self.text, self.mistakeID]

        except IndexError:
            self.mistakeID = '-'
            return [self.text, self.mistakeID]

        except AttributeError:
            self.mistakeID = '-'
            return [self.text, self.mistakeID]
