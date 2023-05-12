import pandas as pd
from sentence_transformers import SentenceTransformer

import visualization
from bertopic import BERTopic

from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer


# 读取文件
df = pd.read_excel('file.xlsx', header=None, names=['Abstract', 'Date'], skiprows=[0])


# 删除空值
df = df.dropna()

# 将Abstract列的内容保存至docs变量中
docs = df['Abstract'].tolist()
timestamp = df['Date'].tolist()

# 尝试加载模型
try:
    print("Loading  model...")
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=True)
    topic_model = BERTopic.load('digital_twin')
except:
    # 模型不存在，重新训练
    print("Model not found, training model...")
    # 停用词
    vectorizer_model = CountVectorizer(stop_words="english")

    # 常用词
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=True)

    topic_model = BERTopic(verbose=True, nr_topics="auto", vectorizer_model=vectorizer_model, ctfidf_model=ctfidf_model)

    topic_model.fit_transform(docs, embeddings)

    topic_model.save("digital_twin")

# all_topics = topic_model.get_topics()
# document_info = topic_model.get_document_info(docs)
# 保存document_info为xlsx文件
# document_info.to_excel("document_info.xlsx")


# 保存all_topics为xlsx文件
# visualization.save_topics(all_topics)
#
# visualization.save_topics(topic_model.get_topic_info())



# 可视化主题
print("Visualizing topics...")
visualization.visualize_topics_hierarchy_barchart_heatmap_rank(topic_model, docs,len(topic_model.get_topics()))
#
# # 可视化文档
visualization.visualize_documents(topic_model, docs, embeddings)
#
# # 可视化主题随时间变化
# # 需要仅含Abstract和Date的csv文件
visualization.visualize_topics_over_time(topic_model, docs,timestamps=timestamp)
