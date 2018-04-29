import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from bokeh.charts import Donut, show, output_file
from bokeh.charts import defaults
from bokeh.layouts import column
defaults.width = 500
defaults.height = 500


data1 = pd.read_csv(r"train.csv")

data2 = pd.read_csv(r'latex.csv')
df = pd.DataFrame(columns=(['sentiment']))

x_train = data1["text"]
x_test = data2["How was the event overall ?"]
 
y_train = data1["label"]


vect = CountVectorizer(stop_words='english')

vect.fit(x_train)


xtrain = vect.transform(x_train)
xtest = vect.transform(x_test)

prediction1 = []
prediction2 = []
prediction3 = []
prediction4 = []
prediction5 = []

model1 = MultinomialNB()
model1.fit(xtrain,y_train)
prediction1.append(model1.predict(xtest))

model2 = LogisticRegression()
model2.fit(xtrain,y_train)
prediction2.append(model2.predict(xtest))

model3 = KNeighborsClassifier(n_neighbors=2)
model3.fit(xtrain,y_train)
prediction3.append(model3.predict(xtest))

model4 = RandomForestClassifier()
model4.fit(xtrain,y_train)
prediction4.append(model4.predict(xtest))

model5 = AdaBoostClassifier()
model5.fit(xtrain,y_train)
prediction5.append(model5.predict(xtest))


pcount = 0
ncount = 0
for i in range(0,len(data2)):
    p = 0
    n = 0
    if prediction1[0][i] == "pos":
        p = p + 1
    else:
        n = n + 1  
        
    if prediction2[0][i] == "pos":
        p = p + 1
    else:
        n = n + 1   
        
    if prediction3[0][i] == "pos":
        p = p + 1
    else:
        n = n + 1 
        
    if prediction4[0][i] == "pos":
        p = p + 1
    else:
        n = n + 1
        
    if prediction5[0][i] == "pos":
        p = p + 1
    else:
        n = n + 1    
        
    if p>n:
        pcount = pcount + 1
        df1 = pd.DataFrame([{'sentiment':'Positive'}])
        df = df.append(df1)
    else:
        ncount = ncount + 1
        df1 = pd.DataFrame([{'sentiment':'Negative'}])
        df = df.append(df1)

p1 = Donut(data2, label=['Did the volunteers manage the event properly ?'], title="Did the volunteers manage the event properly ?",text_font_size='12pt')
p2 = Donut(data2, label=['How was the speaker ?'], title="How was the speaker ?",text_font_size='12pt')
p3 = Donut(data2, label=['Was the event lengthy ?'], title="Was the event lengthy ?",text_font_size='12pt')
p4 = Donut(df, label=['sentiment'], title="How was the event overall ?",text_font_size='12pt')


output_file("chart.html")

show(column(p1,p2,p3,p4))