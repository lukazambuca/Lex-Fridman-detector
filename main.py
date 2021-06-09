import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim.models.phrases import Phrases, Phraser
import multiprocessing

from gensim.models import Word2Vec

###############################################
#      Logistic regression model              #
###############################################


df = pd.read_csv("clean_df.csv")
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['speaker'])

vectorizer = CountVectorizer()
vectorizer.fit_transform(df['clean_text'].values.astype('U'))

y = df['target'].values
X = vectorizer.transform(df['clean_text'].values.astype('U'))

classifier = LogisticRegression(C = 0.8, penalty='l2')
classifier.fit(X, y)


#####################################################
#                  Gensim                           #
#####################################################

#load the saved model
model = gensim.models.Word2Vec.load("Lex.model")
vectors = model.wv

ls =[]

def similar_text(word):

    if len(word.split()) > 1:
        return "Please enter a single word"

    if word in model.wv.key_to_index:
        output = vectors.most_similar(word)
        for i in range(0, len(output)):
            similar_words = output[i][0]
            ls.append(similar_words)
            #print(output[i][0])
        return ls
    else:
        return "This word does not appear in the podcast. Please try another word."

###############################################
#            Prediction function               #
###############################################

def text_predict(my_text):
    doc = list(my_text.split(" "))
    doc = vectorizer.transform(doc)
    predicted = classifier.predict(doc)
    if predicted[0] == 1:
        return "This is something Lex would say"
    else:
        return "Not something Lex would say"

def main():
    st.title('The Lex Fridman Detector')
    st.write("Lex Fridman is a computer scientist, guitar enthuisast, jiu jistsu practioner and podcaster. This just goes to show how advanced"
                 " the latest robots have become. His recent software update has allowed him to smile at jokes and even recite Dostoevsky."
                 " In this project we use machine learning to predict whether or not a certain phrase is something that Lex Fridamn would say.")
    #image = Image.open('fridmancover.webp')
    #st.image(image)
    st.video('https://www.youtube.com/watch?v=ZVbAOpYv7Sk')
    #st.write(score)


    message1 = st.text_area("Please Enter Lex Fridman Phrase", "Type Here")
    if st.button("Detect"):
        result1 = text_predict(message1)
        st.success(result1)

    st.write("Using word embeddings we can also suggest similar words within Lex Fridman's conversations to the word being entered below. "
            "The model finds words which are quite similar in meaning or are at least related to the word being subjected to the prediction model. For example, type in 'russia'"
             " and see what comes out.")

    word = st.text_input("Please enter a lowercase word","Type Here")
    if st.button("Find Similar"):
        result2 = similar_text(word)
        st.success(result2)


    #st.write(output)
    st.subheader("How does it work?")
    st.write("In short, we use the bag of words representation technique to reduce dialogue text to a vector that is "
             "then used in a simple logistic regression machine learning model. "
             "The logistic regression model is one of the most well known models and is often applied to text classification problems due to its simplicty and interpretability. "
             " The model gives the probability of a certain class, which is usually dichotomous outcome variables that are often labeled 0 and 1. The model was compared against other"
             " popular models such as naive bayes, k-means and convultional neural networks. The logistic regression model achieved the highest accuracy with a score of 77% (train-test split: 75/25)."
        " The training data was obtained from Lex Fridman podcast dialogues that were initially downloaded in PDF format from the site Happy Scribes and then converted into a text format. "
        "The data then undergoes a exhaustive data preprocessing stage where we remove stop words, lemmatize the text and much more. At this moment in time the data"
             " only consists of two labelled conversations: one with Joscha Bach and another with Andrew Huberman."
        " Below is a snapshot of the data that goes into the machine learning algorithm. ")

    df[['speaker','clean_text']]

    st.write("Thereafter we encode the target variable, which is the speaker. This is treated as a binary classification task as we are trying to "
             "predict if the speaker is Lex Fridman or his guest (0 or 1).")

    st.subheader("Understanding Bag of Words")
    st.write("The main idea behind the bag of words technique is to convert text into vectors. "
             "It does so by learning a vocabulary from all of the documents, then models each document by counting the number of times each word appears. "
             "For example, let us consider the following two sentences:")
    st.write("Sentence 1: Thank you for joining the podcast")
    st.write("Sentence 2: It is a pleasure to be here")
    example = pd.DataFrame({
        'first sentence': ["Thank", "you", "for", "joining", "the", "podcast", 0],
        'second sentence': ["It", "is", "a", "pleasure","to","be","here"]
    })
    st.write("The two sentences can also be represented in the following way: ")

    example

    st.write("To obtain our bags of words we go through all the words in the above sentences and "
             "count the presence of each word and mark 0 for absence. This is called the scoring method.")
    st.write("The scoring of the first sentence would look as follows")

    frequency = pd.DataFrame({
        'Word': ["Thank", "you", "for", "joining", "the", "podcast","It", "is", "a", "pleasure", "to", "be", "here"],
        'Frequency': [1,1,1,1,1,1,0,0,0,0,0,0,0]
    })
    frequency

    st.write("The same can be done for the second sentence. Putting it all together we get")

    cars = {'Thank': [1,0],
            'you': [1,0],
            'for': [1, 0],
            'joining': [1, 0],
            'the': [1, 0],
            'podcast': [1, 0],
            'It': [0, 1],
            'is': [0, 1],
            'a': [0, 1],
            'pleasure': [0, 1],
            'to': [0, 1],
            'be': [0, 1],
            'here': [0, 1],


            }

    final = pd.DataFrame(cars, columns=['Thank', 'you','for','joining','joining','the','podcast','It','is','a','pleasure',
                                        'to','be','here'], index=['Sentence 1', 'Sentence 2'])

    final

    st.write("Unfortunately, the bag of words approach has some noticeable disadvantages. The semantic analysis of the sentence is not taken into consideration."
             " Furthermore, the word arrangement is discarded, which means that the arrangement of words in the sentence does not matter in bag of words techniques. "
             "For example, in the bag of words techniques, the sentence “Red means stop” is represented the same way as “Stop means read” which of course is incorrect.  "
             " If you want to learn more about the bag of words technique and how it is used in machine learning models, please check out "
             "this [link](https://www.youtube.com/watch?v=UFtXy0KRxVI).")

    #image2 = Image.open('cmlexfridman.png')
    #st.image(image2)

    st.subheader("How did we find similar words?")

    st.write("To find similar words we used Word2vec, which is a technique that is used to produce word embeddings for better word representation. "
             "A word embedding is a word representation type that allows machine learning algorithms to understand words with similar meanings. "
             "The Word2vec model represents words in vector space representation and words are represented in the form of vectors and placement is"
             " done in such a way that similar meaning words appear together and dissimilar words are located far away. "
             "Stanford University offers a great lecture on the topic of Word2vec, which is accessible [here](https://www.youtube.com/watch?v=ERibwqs9p38). ")

    st.subheader("Discussion")
    st.write("Why predict the speaker? I was interested to find out if a persons vocabulary was unique enough to be modelled."
             " Luckily I had access to my favourite podcasters labeled transcripts. This allowed me to transform a simple conversation "
             "into a text classification problem. If the results appear to be promising then the work here could be extended (quality data permitting) to "
             "classifying speakers solely on textual data. Ultimately answering the question, is this something that 'xyz' would say? It may not seem like " 
             "it but I believe this a useful pursuit with the advancements (and ease of use) of deep fakes. In the future deep fakes will allow people to" 
            " construct artificial voice overs, which poses a great security risk. Other than battling deep fakes, speaker classification can"
             " hopefully automate the labelling process of transcripts. Out of the hundreds of conversations that Lex Fridman had, only two were"
             " labelled (conversations with Joscha Bach and Andrew Huberman). Labelled in the sense that at each timestamp it was classified who the speaker was, "
             "either Lex Fridman or the guest. ")





if __name__ == '__main__':
    main()
