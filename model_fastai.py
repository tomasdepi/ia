from fastai.tabular import *
import pandas as pd

df = pd.read_csv('/home/depi/Desktop/ia/all_languages.csv')
dataframeLen = len(df)
lenForTesting = round(dataframeLen * 0.25)
dfTesting = df.sample(lenForTesting)
dfTraining = df[~df.index.isin(list(dfTesting.index))]

outputColumn = 'Language'
inputColumns = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19','c20']
cont_names = []
procs = [FillMissing, Categorify, Normalize]

test = TabularList.from_df(dfTraining.copy(), cat_names=inputColumns, cont_names=cont_names)


data = (TabularList.from_df(dfTraining, cat_names=inputColumns, cont_names=cont_names, procs=procs)
                           .split_by_idx(list(range(800,1000)))
                           .label_from_df(cols=outputColumn)
                           .add_test(test)
                           .databunch())


learn = tabular_learner(data, layers=[200,100], metrics=accuracy)
learn.fit(5, 1e-1)

lr_find(learn)

dfResult = pd.DataFrame(columns=["Word", "Language", "Input", "Expected Result", "Result", "Match"])

for rowIndex, row in dfTesting.iterrows():

    input = tuple(row[cat_names])
    languageString = constants.dicLanguageIdToString[row["Language"]]
    languageId = row["Language"]
    result = str(learn.predict(row)[0])
    match = result==str(languageId)

    rowToSave = [row["Word"],languageString,input, languageId, result, match]
    
    print(rowToSave)
    dfResult.loc[rowIndex] = rowToSave
    