# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import time

import openai
import pandas as pd
from openai.error import RateLimitError

openai.api_key = 'xxx'

def request_openai(prompt, attempts=5, delay=60):
    for i in range(attempts):
        try:
            response = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
            return response
        except RateLimitError:
            if i < attempts - 1:  # i is zero indexed
                time.sleep(delay)  # wait for delay seconds before trying again
                continue
            else:
                raise

data = pd.read_excel('../document_info.xlsx', header=0, index_col=0)

print(data.columns)
documents = [''] * 249
keywords = [''] * 249
for index, item in data.iterrows():

    if int(item['Topic']) != -1 and item['Representative_document']:

        documents[int(item['Topic'])] += '---' + item['Document'] + '\n'

        if keywords[int(item['Topic'])] == '':
            keywords[int(item['Topic'])] += item['Name']

print(documents)

dataframe = pd.DataFrame(columns=['response'])

dataframe = pd.DataFrame(columns=['response'])

for i in range(249):
    prompt = """
    I have topic that contains the following documents: %s\n
    The topic is described by the following keywords: %s
    
    Based on the information above, extract a short topic label in the following format:
    topic: <topic label>
    """ % (documents[i], keywords[i])
    print(prompt)
    response = request_openai(prompt)
    # 存储response
    dataframe.loc[i] = response['choices'][0]['message']['content']
    print(response['choices'][0]['message']['content'])

# 存储dataframe为xlsx文件
dataframe.to_excel('topic_label.xlsx')

