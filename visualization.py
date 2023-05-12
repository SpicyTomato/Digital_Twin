import pandas as pd


def visualize_topics_hierarchy_barchart_heatmap_rank(topic_model, docs,topicsNum):

    # 可视化主题
    print("Visualizing topics...")
    fig = topic_model.visualize_topics()
    fig.write_html("visualize_topics.html")

    # 主题树
    print("Visualizing hierarchy...")
    hierarchical_topics = topic_model.hierarchical_topics(docs)

    fig_2 = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
    fig_2.write_html("visualize_hierarchy.html")

    tree = topic_model.get_topic_tree(hierarchical_topics)

    # tree为string格式，保存为markdown文件
    with open("topic_tree.md", "w", encoding="utf-8") as f:
        f.write(tree)

    # 主题柱状图
    print("Visualizing barchart...")
    fig = topic_model.visualize_barchart(top_n_topics=topicsNum)
    fig.write_html("visualize_barchart.html")

    # 主题热力图
    print("Visualizing heatmap...")
    fig = topic_model.visualize_heatmap()
    fig.write_html("visualize_heatmap.html")

    # 主题分数
    print("Visualizing term rank...")
    fig = topic_model.visualize_term_rank()
    fig.write_html("visualize_term_rank.html")


def visualize_documents(topic_model, docs,embedding):


    fig = topic_model.visualize_documents(docs, reduced_embeddings=embedding)
    fig.write_html("visualize_documents.html")


def probablitiesandDistribution(topic_model, docs):
    topics, probs = topic_model.fit_transform(docs)

    # Calculate the topic distributions on a token-level
    topic_distr, topic_token_distr = topic_model.approximate_distribution(docs, calculate_tokens=True)

    # To visualize the probabilities of topic assignment
    fig = topic_model.visualize_distribution(probs[0])
    fig.write_html("visualize_probabilities.html")

    # To visualize the topic distributions in a document
    fig = topic_model.visualize_distribution(topic_distr[0])
    fig.write_html("visualize_distribution.html")

    # Visualize the token-level distributions
    df = topic_model.visualize_approximate_distribution(docs, topic_token_distribution=topic_token_distr)

    # 保存df

    df.to_csv("visualize_approximate_distribution.csv", index=False)
    df


def visualize_topics_over_time(topic_model,abstract,timestamps):

    topics_over_time = topic_model.topics_over_time(abstract, timestamps)

    # Visualize topics over time
    fig = topic_model.visualize_topics_over_time(topics_over_time)
    fig.write_html("visualize_topics_over_time.html")


def generateTopic(topic_model, docs):
    print("Saving......")
    topics, probs = topic_model.fit_transform(docs)

    print(topics)
    print(probs)

    # 保存topics和probs
    df = pd.DataFrame({'topics': topics, 'probs': probs})
    df.to_csv('topics_probs.csv', index=False)
    return None


def save_topics(all_topics):
    print("Saving......")
    df = pd.DataFrame({'topics': all_topics})
    df.to_csv('topics.csv', index=False)
    return None
