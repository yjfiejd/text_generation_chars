# -*- coding:utf8 -*-
# @TIME : 2018/5/2 下午10:17
# @Author : Allen
# @File : rnn_text_generation_words.py

import os
import  numpy as np
import nltk
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import  Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from gensim.models.word2vec import Word2Vec

#【1】 读取数据
os.chdir('/Users/a1/Desktop/算法实战/kaggel_06/Char_rnn/')
raw_text = open('Project_Gutenberg_Complete_Works_of_Winston_Churchill的副本.txt').read()
raw_text = raw_text.lower()
sentensor = nltk.data.load('tokenizers/punkt/english.pickle')
sents = sentensor.tokenize(raw_text) #把文章变为句子
corpus = [nltk.word_tokenize(x) for x in sents]
# for sen in sents:
#     corpus.append(nltk.word_tokenize(sen)) #把句子切分成单词
print(len(sents))
print(type(sents))
print(sents[:2])
print('**************')
print(corpus[:2])
print(len(corpus))
print(type(corpus))

#输出结果：
# 13227
# <class 'list'>
# ['the project gutenberg ebook of the complete pg edition of the works of\nwinston churchill, by winston churchill\n[the author is the american winston churchill not the british]\n\nthis ebook is for the use of anyone anywhere at no cost and with\nalmost no restrictions whatsoever.', "you may copy it, give it away or\nre-use it under the terms of the project gutenberg license included\nwith this ebook or online at www.gutenberg.net\n\n\ntitle: the complete pg edition of the works of winston churchill\n\nauthor: winston churchill\n\nrelease date: october 19, 2004 [ebook #5400]\n[last updated: july 16, 2011]\n\nlanguage: english\n\n\n*** start of this project gutenberg ebook works of winston churchill ***\n\n\n\n\nproduced by david widger\n\n\n\n\n\nthe complete pg edition of the works of winston churchill\n\nby winston churchill\n\n[the author is the american winston churchill not the british]\n\n\n\ncontents:\n\n     the crossing\n     the dwelling place of light\n     mr. crewe's career\n     a far country\n     coniston\n     the inside of the cup\n     richard carvel\n     a modern chronicle\n     the celebrity\n     the crisis\n     dr. jonathan (play)\n     a traveller in wartime\n     an essay on the american contribution and the democratic idea\n\n\n\n\n\nthe crossing\n\nby winston churchill\n\n\n\ncontents\n\nbook i. the borderland\n\ni.      the blue wall\nii."]
# **************
# [['the', 'project', 'gutenberg', 'ebook', 'of', 'the', 'complete', 'pg', 'edition', 'of', 'the', 'works', 'of', 'winston', 'churchill', ',', 'by', 'winston', 'churchill', '[', 'the', 'author', 'is', 'the', 'american', 'winston', 'churchill', 'not', 'the', 'british', ']', 'this', 'ebook', 'is', 'for', 'the', 'use', 'of', 'anyone', 'anywhere', 'at', 'no', 'cost', 'and', 'with', 'almost', 'no', 'restrictions', 'whatsoever', '.'], ['you', 'may', 'copy', 'it', ',', 'give', 'it', 'away', 'or', 're-use', 'it', 'under', 'the', 'terms', 'of', 'the', 'project', 'gutenberg', 'license', 'included', 'with', 'this', 'ebook', 'or', 'online', 'at', 'www.gutenberg.net', 'title', ':', 'the', 'complete', 'pg', 'edition', 'of', 'the', 'works', 'of', 'winston', 'churchill', 'author', ':', 'winston', 'churchill', 'release', 'date', ':', 'october', '19', ',', '2004', '[', 'ebook', '#', '5400', ']', '[', 'last', 'updated', ':', 'july', '16', ',', '2011', ']', 'language', ':', 'english', '***', 'start', 'of', 'this', 'project', 'gutenberg', 'ebook', 'works', 'of', 'winston', 'churchill', '***', 'produced', 'by', 'david', 'widger', 'the', 'complete', 'pg', 'edition', 'of', 'the', 'works', 'of', 'winston', 'churchill', 'by', 'winston', 'churchill', '[', 'the', 'author', 'is', 'the', 'american', 'winston', 'churchill', 'not', 'the', 'british', ']', 'contents', ':', 'the', 'crossing', 'the', 'dwelling', 'place', 'of', 'light', 'mr.', 'crewe', "'s", 'career', 'a', 'far', 'country', 'coniston', 'the', 'inside', 'of', 'the', 'cup', 'richard', 'carvel', 'a', 'modern', 'chronicle', 'the', 'celebrity', 'the', 'crisis', 'dr.', 'jonathan', '(', 'play', ')', 'a', 'traveller', 'in', 'wartime', 'an', 'essay', 'on', 'the', 'american', 'contribution', 'and', 'the', 'democratic', 'idea', 'the', 'crossing', 'by', 'winston', 'churchill', 'contents', 'book', 'i.', 'the', 'borderland', 'i.', 'the', 'blue', 'wall', 'ii', '.']]
# 13227s
# <class 'list'>

#把语料库用Word2Vec表示为词向量
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# stop = stopwords.words('english') #去除停用词
# wordnet_lemmatizer = WordNetLemmatizer() #提取词干

w2v_model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=4)
print(w2v_model['complete'])

# #输出结果：
# [-0.01046881 -0.04057425  0.0057188   0.0116123  -0.05806362  0.00096401
#   0.00570232 -0.030064    0.02195651 -0.00773561  0.02267634  0.01824553
#  -0.04346155 -0.01898313 -0.02093714 -0.01256765  0.01893479 -0.00446839
#   0.0292802   0.05611555 -0.06105915 -0.00869842 -0.03651512  0.04331645
#  -0.02025064  0.03868391  0.02032129 -0.01346867  0.00702124 -0.00383223
#   0.02073413 -0.01883716  0.04261814  0.02174589  0.00462364 -0.00049488
#   0.00556233  0.01914215  0.01060296 -0.02695866  0.03043551 -0.01215882
#   0.03256044  0.0171234  -0.02953585 -0.01218379 -0.00388132 -0.01770512
#   0.03625446  0.03452534 -0.02677291  0.01956664  0.02102937 -0.02325316
#   0.02132765  0.01637649 -0.00129769  0.03993525 -0.04866015 -0.01714137
#   0.00657561 -0.01503081 -0.00371172 -0.00724585  0.02551177 -0.01947266
#  -0.02426686 -0.02263844  0.02851453 -0.04807662  0.01277628  0.0113344
#  -0.00925684 -0.02168215 -0.00213529 -0.00540714 -0.0063272   0.00641162
#   0.03273816  0.00484689  0.01976413 -0.04357935  0.00377653 -0.00318833
#  -0.01364405  0.02580834  0.02591778  0.01076849  0.01548144 -0.00801603
#  -0.01266945  0.0440975   0.02654843 -0.00344783  0.01449298 -0.02398393
#   0.0195865   0.01989166 -0.00507161  0.02223123  0.01879907 -0.06201274
#   0.00265529 -0.00876945  0.01293628  0.02235157 -0.00607967  0.01312169
#   0.05216238  0.00920464  0.03294037 -0.01929664  0.03317495 -0.00782996
#   0.00292527  0.01666457 -0.07025924 -0.03182978  0.01647154 -0.00256438
#  -0.01808942 -0.02077844 -0.04657688  0.0242828  -0.00879178 -0.00198684
#  -0.01038997  0.02082653]

#接下来，其实我们还是以之前的方式来处理我们的training data，把源数据变成一个长长的x，好让LSTM学会predict下一个单词：
raw_input = [item for sublist in corpus for item in sublist]
print(len(raw_input))
print(raw_input[:10])
#输出：
# 235236
# ['the', 'project', 'gutenberg', 'ebook', 'of', 'the', 'complete', 'pg', 'edition', 'of']

#这里本来想输出所有的词向量集合
text_stream = []
vocab = w2v_model.vocabulary
print(vocab)
for word in raw_input:
    if word in vocab:
        text_stream.append(word)
print(len(text_stream))
# 报错：TypeError: argument of type 'Word2VecVocab' is not iterable


#【2】构造训练测试集合
seq_length = 10
x = []
y = []
for i in range(0, len(text_stream) - seq_length):

    given = text_stream[i:i + seq_length]
    predict = text_stream[i + seq_length]
    x.append(np.array([w2v_model[word] for word in given]))
    y.append(w2v_model[predict])

x = np.reshape(x, (-1, seq_length, 128))
y = np.reshape(y, (-1,128))

#接下来我们做两件事：
# 我们已经有了一个input的数字表达（w2v），我们要把它变成LSTM需要的数组格式： [样本数，时间步伐，特征]
# 第二，对于output，我们直接用128维的输出

#【3】构造模型
model = Sequential()
model.add(LSTM(256, dropout_W=0.2, dropout_U=0.2, input_shape=(seq_length, 128)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam')
#跑模型
model.fit(x, y, nb_epoch=50, batch_size=4096)

#【4】结果
def predict_next(input_array):
    x = np.reshape(input_array, (-1,seq_length,128))
    y = model.predict(x)
    return y

def string_to_index(raw_input):
    raw_input = raw_input.lower()
    input_stream = nltk.word_tokenize(raw_input)
    res = []
    for word in input_stream[(len(input_stream)-seq_length):]:
        res.append(w2v_model[word])
    return res

def y_to_word(y):
    word = w2v_model.most_similar(positive=y, topn=1)
    return word

def generate_article(init, rounds=30):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_word(predict_next(string_to_index(in_string)))
        in_string += ' ' + n[0][0]
    return in_string

init = 'Language Models allow us to measure how likely a sentence is, which is an important for Machine'
article = generate_article(init)
print(article)