import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# 读取Excel文件
file = '../file.xlsx'
df = pd.read_excel(file, engine='openpyxl')

# 数据预处理
nltk.download('punkt')  # 新增：下载punkt资源
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    if not isinstance(text, str):  # 检查输入是否为字符串
        return []  # 如果不是字符串，返回空列表

    tokens = nltk.word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return words


df['Processed_Abstract'] = df['Abstract'].apply(preprocess)

# 构建文档-词汇矩阵
dictionary = Dictionary(df['Processed_Abstract'])
corpus = [dictionary.doc2bow(text) for text in df['Processed_Abstract']]

# LDA建模
num_topics = 5  # 选择主题数量
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, random_state=42)

# 可视化主题模型
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis_data)
pyLDAvis.save_html(vis_data, 'lda_visualization.html')

from gensim.models import CoherenceModel

coherence_model = CoherenceModel(model=lda_model, texts=df['Processed_Abstract'], dictionary=dictionary,
                                 coherence='c_v')
coherence_score = coherence_model.get_coherence()
print(f'Topic Coherence: {coherence_score}')


def get_topic_keywords(lda_model, top_n=10):
    topic_keywords = []
    for topic_id, topic_terms in lda_model.show_topics(num_topics=num_topics, num_words=top_n, formatted=False):
        keywords = [term for term, _ in topic_terms]
        topic_keywords.append(keywords)
    return topic_keywords


topic_keywords = get_topic_keywords(lda_model, top_n=10)

from sklearn.metrics import jaccard_score


def topic_diversity(topic_keywords):
    n_topics = len(topic_keywords)
    jaccard_similarities = []

    for i in range(n_topics):
        for j in range(i + 1, n_topics):
            jaccard_sim = jaccard_score(topic_keywords[i], topic_keywords[j], average='weighted')
            jaccard_similarities.append(jaccard_sim)

    diversity_score = 1 - sum(jaccard_similarities) / len(jaccard_similarities)
    return diversity_score


diversity_score = topic_diversity(topic_keywords)
print(f'Topic Diversity: {diversity_score}')
