import pandas as pd
from gensim import corpora, models
import nltk
import string
import matplotlib.pyplot as plt

ted_main_data = pd.read_csv('./ted-talks/ted_main.csv')
ted_main_data.head()

ted_transcript_data = pd.read_csv('./ted-talks/transcripts.csv')
ted_transcript_data.head()

ted_total_data = pd.concat([ted_main_data, ted_transcript_data], axis=1, join='inner')
ted_total_data.shape
ted_total_data.head()

useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)

speakers = ted_total_data['speaker_occupation'].dropna()
speakers_list = speakers.tolist()
speakers_token = []
for e in speakers_list: 
    temp = nltk.word_tokenize(e)
    speakers_token.append(temp)

useful_token = []
for e_list in speakers_token:
    for e in e_list:
        if e not in useless_words:
            useful_token.append(e)

dic = corpora.Dictionary([useful_token])
corpus = [dic.doc2bow(text) for text in [useful_token]]
import gensim
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dic, num_topics=15)

lda.show_topics(15)

speakers = speakers.dropna()
ted_total_data = ted_total_data.dropna()

category_list = ['writer', 'expert', 'researcher', 'expert', 'scientist', 'entrepreneur', 'designer', 'artist', 'journalist', 'biologist', 'author', 'activist', 'advocate']

refer = pd.Series()
index = 0
for s in speakers:
    flag = 0
    for c in category_list:
        if c in s.lower():
            refer.set_value(index, c)
            index+=1
            flag = 1
            break
    if flag == 0:
        refer.set_value(index, 'others')
        index+=1

refer.name = 'speaker_category'
ted_total_data = ted_total_data.join(refer)
ted_total_data.head()

extract_data = ted_total_data[['speaker_category', 'views', 'comments']]
views_sort = extract_data.groupby(['speaker_category'], as_index=False).mean().sort_values(['views'], ascending=False)

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 13}

matplotlib.rc('font', **font)

plt.title('Average Views of Different Speaker Categories', fontsize=50)
plt.bar(views_sort['speaker_category'], views_sort['views'])
plt.ylabel('Views', fontsize=20)
plt.xlabel('Speaker Categories', fontsize=20)
plt.rcParams['figure.figsize'] = ('25,18')
