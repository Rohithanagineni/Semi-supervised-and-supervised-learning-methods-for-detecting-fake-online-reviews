from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from gensim import matutils,corpora, models
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

main = tkinter.Tk()
main.title("SEMI-SUPERVISED AND SUPERVISED LEARNING METHODS FOR DETECTING FAKE ONLINE REVIEWS")
main.geometry("1300x1200")

def createdf(root_dir,review):
    listOfFiles = []
    for (dirpath, dirnames, filenames) in os.walk(root_dir):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    labeled_class = []
    reviews = []
    actual_class =[]
    for j in listOfFiles:
        labeled_class.append(review)
        k = str(open(j,encoding='utf-8').read())
        reviews.append(k)
        actual_class.append(str(j.split('\\')[1].split('_')[0]))
    data = pd.DataFrame({'labeled_class':labeled_class,'review':reviews,'actual_class':actual_class})
    return data

def importdata():
    global negative_df,positive_df
    text.delete('1.0',END)
    negative_df = createdf('negative_reviews','negative')
    text.insert(END, "No of Rows: "+str(negative_df.shape[0])+"\nNo of Columns:"+str(negative_df.shape[1])+"\n")
    positive_df = createdf('positive_reviews','positive')
    text.insert(END, "No of Rows: "+str(negative_df.shape[0])+"\nNo of Columns:"+str(negative_df.shape[1])+"\n")
    text.insert(END,"Actual class Information for negative: "+str(negative_df['actual_class'].value_counts())+"\n")
    text.insert(END,"Labelled class Information for negative: "+str(negative_df['labeled_class'].value_counts())+"\n")
    text.insert(END,"Actual class Information for Positive: "+str(positive_df['actual_class'].value_counts())+"\n")
    text.insert(END,"Labelled class Information for Posituve: "+str(positive_df['labeled_class'].value_counts())+"\n")

    
def preprocess():
    text.delete('1.0',END)
    global data
    target = []
    for i in positive_df.index:
        if ((positive_df['labeled_class'][i] == 'positive') & (positive_df['actual_class'][i] == 'truthful')):
            target.append(2)
        elif ((positive_df['labeled_class'][i] == 'positive') & (positive_df['actual_class'][i] == 'deceptive')):
            target.append(1)
        else:
            print('Error!')
    positive_df['target'] = target

    target = []
    for i in negative_df.index:
        if ((negative_df['labeled_class'][i] == 'negative') & (negative_df['actual_class'][i] == 'truthful')):
            target.append(3)
        elif ((negative_df['labeled_class'][i] == 'negative') & (negative_df['actual_class'][i] == 'deceptive')):
            target.append(4)
        else:
            print('Error!')
    negative_df['target'] = target

    data = positive_df.merge(negative_df,how='outer')
    data = data[['review','target']]
    text.insert(END,"Basic information of Data: \n"+str(data.head())+"\n")
    text.insert(END,"Target variable information: \n"+str(data.target.value_counts())+"\n")

def extract_tokens(df):
    review_tokenized = []
    lmt = WordNetLemmatizer()
    for index, datapoint in df.iterrows():
        tokenize_words = word_tokenize(datapoint["review"].lower(),language='english')
        pos_word = pos_tag(tokenize_words)
        tokenize_words = ["_".join([lmt.lemmatize(i[0]),i[1]]) for i in pos_word if (i[0] not in stopwords.words("english") and len(i[0]) > 2)]
        token = ' '.join(tokenize_words)
        review_tokenized.append(token)
    df["review_tokenized"] = review_tokenized
    return df

def tfidf():
    text.delete('1.0', END)
    global x, data, tfidf
    global X_train, X_test, y_train, y_test
    data = extract_tokens(data)
    print(data.head())

    tfidf = TfidfVectorizer( lowercase=False,stop_words='english',max_features=3000)
    x = tfidf.fit_transform(data['review_tokenized'])
    text.insert(END,"Shape of Data after TFIDF: "+str(x.shape)+"\n")
    text.insert(END,"Data Information: \n"+str(x)+"\n")
    text.insert(END,"Spliting the Extracted features from TFIDF to Train and Test\n")
    X_train, X_test, y_train, y_test = model_selection.train_test_split(x, data["target"], test_size=0.3, random_state=2016)
    text.insert(END,"Shape of train Data after Split: "+str(X_train.shape)+"\n")
    text.insert(END,"Shape of test Data after Split: "+str(X_test.shape)+"\n")

def ouput(pred):
    if pred == 1.0 :
        return('Fake Review (Positive)')
    elif pred == 2.0:
        return('True Review (Positive)')
    elif pred == 3.0:
        return('True Review (Negative)')
    else :
        return('Fake Review (Negative)')

def predict(model, modelName):
    with open('test.txt') as file:
        data = file.read()
        review_tokenized = []
        print(data)
        lmt = WordNetLemmatizer()
        tokenize_words = word_tokenize(data.lower(),language='english')
        print(tokenize_words)
        pos_word = pos_tag(tokenize_words)
        tokenize_words = ["_".join([lmt.lemmatize(i[0]),i[1]]) for i in pos_word if (i[0] not in stopwords.words("english") and len(i[0]) > 2)]
        token = ' '.join(tokenize_words)
        review_tokenized.append(token)
        res = tfidf.transform(review_tokenized)
        res = res.toarray() 
        pred = model.predict(res)
        review = ouput(pred)
        text.insert(END,"Predicted Review for test Data is : "+str(review)+" using "+str(modelName)+"\n")


def rfc():
    text.delete('1.0', END)
    global rfc,rfc_acc
    rfc = RandomForestClassifier(min_samples_leaf= 1, min_samples_split= 2, n_estimators= 100, verbose=1,n_jobs=-1)
    rfc.fit(X_train, y_train)
    
    text.insert(END,"Train data accuracy of RandomForest: "+str(rfc.score(X_train,y_train))+"\n")
    text.insert(END,"Test data accuracy of RandomForest: "+str(rfc.score(X_train,y_train))+"\n")
    rfc_acc = rfc.score(X_test,y_test)
    predict(rfc,"RandomRorest")

def knn():
    global knn,knn_acc
    knn = KNeighborsClassifier(n_neighbors=21)
    knn.fit(X_train,y_train)
    text.insert(END,"Train data accuracy of KNN: "+str(knn.score(X_train,y_train))+"\n")
    text.insert(END,"Test data accuracy of KNN: "+str(knn.score(X_train,y_train))+"\n")
    knn_acc = knn.score(X_test,y_test)
    predict(knn,"KNN")
          
def mlp():
    global mlp,mlp_acc
    mlp = MLPClassifier()
    mlp.fit(X_train,y_train)
    text.insert(END,"Train data accuracy of MLP: "+str(mlp.score(X_train,y_train))+"\n")
    text.insert(END,"Test data accuracy of MLP: "+str(mlp.score(X_train,y_train))+"\n")
    mlp_acc = mlp.score(X_test,y_test)
    predict(mlp,"MLP")
    
def semilp():
    global lpm,lpm_acc
    lpm = LabelPropagation()
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(len(y_train)) < 0.3
    labels = np.copy(y_train)
    labels[random_unlabeled_points] = -1
    lpm.fit(X_train.toarray(), labels)
    text.insert(END,"Train data accuracy of SEMI-LP: "+str(lpm.score(X_train,y_train))+"\n")
    text.insert(END,"Test data accuracy of SEMI-LP: "+str(lpm.score(X_train,y_train))+"\n")
    lpm_acc = lpm.score(X_test,y_test)
    predict(lpm,"LabelPropagation")

def semils():
    global lsm, lsm_acc
    lsm = LabelSpreading()
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(len(y_train)) < 0.3
    labels = np.copy(y_train)
    labels[random_unlabeled_points] = -1
    lsm.fit(X_train.toarray(), labels)
    text.insert(END,"Train data accuracy of SEMI-LS: "+str(lsm.score(X_train,y_train))+"\n")
    text.insert(END,"Test data accuracy of SEMI-LS: "+str(lsm.score(X_train,y_train))+"\n")
    lsm_acc = lsm.score(X_test,y_test)
    predict(lsm,"LabelSpreading")

def graph():
    acc = [rfc_acc,knn_acc,mlp_acc,lpm_acc,lsm_acc]
    bars = ('RF','KNN','MLP','LB','LP')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, acc)
    plt.xticks(y_pos, bars)
    plt.show()
    

font = ('times', 16, 'bold')
title = Label(main, text='SEMI-SUPERVISED AND SUPERVISED LEARNING METHODS FOR DETECTING FAKE ONLINE REVIEWS')
title.config(bg='dark salmon', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')

data = Button(main, text="Import Data", command=importdata)
data.place(x=700,y=100)
data.config(font=font1)

process = Button(main, text="Data Pre-Processing", command=preprocess)
process.place(x=700,y=150)
process.config(font=font1)

tfvect = Button(main, text="TF-IDF Vectorize", command=tfidf)
tfvect.place(x=700,y=200)
tfvect.config(font=font1)

rf = Button(main, text="RandomForest Algorithm", command=rfc)
rf.place(x=700,y=250)
rf.config(font=font1)

knn = Button(main, text="KNN Algorithm", command=knn)
knn.place(x=700,y=300)
knn.config(font=font1)

lp = Button(main, text="Label Propagation Algorithm", command=semilp)
lp.place(x=700,y=350)
lp.config(font=font1)

ls = Button(main, text="Label Spreading Algorithm", command=semils)
ls.place(x=700,y=400)
ls.config(font=font1)

nn = Button(main, text="Multi Layer Perceptron", command=mlp)
nn.place(x=700,y=450)
nn.config(font=font1)

gf = Button(main, text="Accuracy Graph", command=graph)
gf.place(x=700,y=500)
gf.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='violet red')
main.mainloop()


    




