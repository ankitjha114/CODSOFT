import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from wordcloud import WordCloud
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer()

df = pd.read_csv('spam.csv', encoding='latin-1')
print(df.sample(5))
print(df.shape)

#Data Cleaning
print(df.info())
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
print(df.sample(5))
df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)
print(df.sample(5))


encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
print(df.head())
print(df.isnull().sum())
print(df.duplicated().sum())
df = df.drop_duplicates(keep='first')
print(df.duplicated().sum())
print(df.shape)

#EDA
print(df.head())
print(df['target'].value_counts())


#print(plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f"), plt.show())
#print(nltk.download('punkt'))
df['num_characters'] = (df['text'].apply(len))
print(df.head())
#print(nltk.download('punkt_tab'))
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
print(df.head())
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
print(df.head())
print(df[['num_characters', 'num_words', 'num_sentences']].describe())
print(df[df['target'] == 0][['num_characters', 'num_words', 'num_sentences']].describe()) #For ham Messages
print(df[df['target'] == 1][['num_characters', 'num_words', 'num_sentences']].describe()) #For spam Messages

print(sns.histplot(df[df['target'] == 0]['num_characters']),plt.show())
print(sns.histplot(df[df['target'] == 1]['num_characters'], color='red'),plt.show())
print(sns.histplot(df[df['target'] == 0]['num_words']), plt.show())
print(sns.histplot(df[df['target'] == 1]['num_words'], color='red'), plt.show())
print(sns.histplot(df[df['target'] == 0]['num_sentences']), plt.show())
print(sns.histplot(df[df['target'] == 1]['num_sentences'], color='red'), plt.show())

print(sns.pairplot(df, hue='target'), plt.show())
print(sns.heatmap(df[['num_characters', 'num_words', 'num_sentences', 'target']].corr(), annot=True), plt.show())

#Data Preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
print(transform_text("Hi Ankit How Are Applying You 20% older?"))
print(df['text'][10])
#print(nltk.download('stopwords'))
print(stopwords.words('english'))
print(string.punctuation)
print(ps.stem("dancing"))
df['transformed_text'] = df['text'].apply(transform_text)
print(df.head())
spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep = " "))
plt.figure(figsize=(15,6))
plt.imshow(spam_wc)
plt.axis('off')
plt.show()
ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep = " "))
plt.figure(figsize=(15,6))
plt.imshow(ham_wc)
plt.axis('off')
plt.show()
print(df.head())

spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

print(spam_corpus)
print(len(spam_corpus))
print(pd.DataFrame(Counter(spam_corpus).most_common(30)))


common_words = Counter(spam_corpus).most_common(30)
df_common = pd.DataFrame(common_words, columns=['word', 'count'])

plt.figure(figsize=(12,6))
sns.barplot(x='count', y='word', data=df_common)

plt.title("Top 30 Words in Spam Messages")
plt.xlabel("Frequency")
plt.ylabel("Words")
# plt.xticks(rotation = 'vertical')
plt.show()

ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)

print(ham_corpus)
print(len(ham_corpus))
print(pd.DataFrame(Counter(ham_corpus).most_common(30)))


common_words = Counter(ham_corpus).most_common(30)
df_common = pd.DataFrame(common_words, columns=['word', 'count'])

plt.figure(figsize=(12,6))
sns.barplot(x='count', y='word', data=df_common)

plt.title("Top 30 Words in ham Messages")
plt.xlabel("Frequency")
plt.ylabel("Words")
# plt.xticks(rotation = 'vertical')
plt.show()

#Model Building
#X = cv.fit_transform(df['transformed_text']).toarray() #For using countvectorizer for training
X = tfidf.fit_transform(df['transformed_text']).toarray()  #For using tfidfvectorizer for training
print(X)
print(X.shape)
y = df['target'].values
print(y)

     #Train Model Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))

mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))

import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))

#Hence the precision, accurancy both good in case of tfid ---> mnb model. hence we chhose mnb to train the model

#checking Other models AlsoFor good comparision
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)

clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}
def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision

train_classifier(svc,X_train,y_train,X_test,y_test)
accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)
print(performance_df)

performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")
print(performance_df1)

sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()

temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)
new_df = performance_df.merge(temp_df,on='Algorithm')
new_df_scaled = new_df.merge(temp_df,on='Algorithm')

temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)
new_df_scaled.merge(temp_df,on='Algorithm')

svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')
voting.fit(X_train,y_train)
y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))

estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()

from sklearn.ensemble import StackingClassifier

clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))



