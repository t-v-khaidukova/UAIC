import re
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from pprint import pprint
import pyLDAvis.gensim
import pickle
import pyLDAvis
import pyLDAvis.gensim_models
import os

# 1. Load Excel
path = "/Users/tatianakhaidukova/Documents/GitHub/NLP2/wikiarticles.xlsx"
papers = pd.read_excel(path)

documents = papers['text'].astype(str).tolist()

# 2. Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return tokens

# Tokenize and preprocess
tokenized_docs = [preprocess(doc) for doc in documents]

# Convert token lists back to strings
clean_docs = [' '.join(tokens) for tokens in tokenized_docs]

# 3. Add clean_docs column to the dataframe
papers['clean_docs'] = clean_docs

# Join all processed texts
long_string = ','.join(papers['clean_docs'].values)

# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

# Generate a word cloud
wordcloud.generate(long_string)

# Visualize the word cloud
wordcloud.to_image()

wordcloud.to_file("/Users/tatianakhaidukova/Documents/GitHub/NLP2/wordcloud.png")

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'also','may','one','use','get','would','could', 'article'])

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]

data = papers['clean_docs'].values.tolist()
data_words = list(sent_to_words(data))

# remove stop words
data_words = remove_stopwords(data_words)

print(data_words[:1][0][:30])

# Create Dictionary
id2word = corpora.Dictionary(data_words)

# Create Corpus
texts = data_words

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1][0][:30])


if __name__ == "__main__":
    # make dictionary and corpus
    id2word = corpora.Dictionary(data_words)
    id2word.filter_extremes(no_below=5, no_above=0.5)
    texts = data_words
    corpus = [id2word.doc2bow(text) for text in texts]

    # launch LDA
    lda_model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=3,
        workers=2,  # число потоков
        random_state=42
    )

    # show top words
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic {idx}: {topic}")

    num_topics = 3  # add the variable 

    os.makedirs('./results', exist_ok=True)

    # Visualize the topics
    LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)

    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(num_topics) +'.html')
