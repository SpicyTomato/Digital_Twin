import pandas as pd

data = pd.read_excel('../document_info.xlsx', header=0, index_col=0)

print(data.columns)
documents = [''] * 37
keywords = [''] * 37
for index, item in data.iterrows():

    if int(item['Topic']) != -1 and item['Representative_document']:

        documents[int(item['Topic'])] += '---' + item['Document'] + '\n'

        if keywords[int(item['Topic'])] == '':
            keywords[int(item['Topic'])] += item['Name']

print(documents[0])

dataframe = pd.DataFrame(columns=['response'])
