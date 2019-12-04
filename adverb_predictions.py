import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import operator
import time

answer={'A tale of a tub.txt': 'Daniel Defoe',
 'All around the moon.txt': 'Jules Verne',
 'Cleopatra.txt': 'Haggard Rider',
 "Eve's Diary.txt": 'Mark Twain',
 'Hard Times.txt': 'Charles Dickens',
 'In the Year 2889.txt': 'Jules Verne',
 'Intentions.txt': 'Oscar Wilde',
 'Middlemarch.txt': 'George Eliot',
 'Oliver Twist.txt': 'Charles Dickens',
 'Rolling Stones.txt': 'O Henry',
 'Roughing It.txt': 'Mark Twain',
 'Sense and Sensibility.txt': 'Jane Austen',
 'The Canterville Ghost.txt': 'Oscar Wilde',
 'The Fortunes and Misfortunes of the Famous Moll Flanders.txt': 'Daniel Defoe',
 'The Gentle Grafter.txt': 'O Henry',
 'The Innocents Abroad.txt': 'Mark Twain',
 'The Life and Adventures of Robinson Crusoe.txt': 'Daniel Defoe',
 'The Lifted Veil.txt': 'George Eliot',
 'The Light That Failed.txt': 'The Light That Failed',
 'The Man.txt': 'Bram Stoker',
 'The Nursery, Alice.txt': 'Lewis Carroll',
 'The people of the mist.txt': 'Haggard Rider',
 'The Turn of the Screw.txt': 'Henry James',
 'Whirligigs.txt': 'O Henry'}

py_path=os.path.dirname(os.path.abspath(__file__))

def normalize_counter(d, target=1.0):
   raw = sum(d.values())
   factor = target/raw
   return Counter({key:value*factor for key,value in d.items()})

def normalize_dict(d, target=1.0):
   raw = sum(d.values())
   factor = target/raw
   return {key:value*factor for key,value in d.items()}

def adverb_fraction(token):
    count_ly=0
    for word in token:
        if word.endswith("ly") and word not in ly_not_adv:
            count_ly+=1
    
    fract=count_ly/len(token)
    return(fract)

f2=open(r"C:\Users\sidch\Desktop\ly_not_adv.txt")
ly_not_adv=f2.read().split()
f2.close()

traindatapath=os.path.join(py_path,"Train Data")
os.chdir(traindatapath)
train_authors=os.listdir(traindatapath)

author_dict={}
adverb_count_author={}
bigram_author_dict={}
trigram_author_dict={}

for author in train_authors:
    all_text_by_author=""
    os.chdir(os.path.join(traindatapath,author))
    all_works=os.listdir()
    print("******************")
    print("Analysing all works by {}".format(author))
    print("-------")
    for work in all_works:
        print("Read '{}' by '{}' succesfully.".format(work,author))
        file=open(work,encoding="utf8")
        text=file.read()
        all_text_by_author=all_text_by_author+text
        file.close()
    token=word_tokenize(all_text_by_author)
    adverb_count_author[author]=adverb_fraction(token)
    
    c=normalize_counter(Counter(token))
    author_dict[author]=dict(c)
    
    bigram = nltk.bigrams(token)
    c2=normalize_counter(Counter(bigram))
    bigram_author_dict[author]=dict(c2)
    
    trigram = nltk.trigrams(token)
    c3=normalize_counter(Counter(trigram))
    trigram_author_dict[author]=dict(c3)
    
    os.chdir(traindatapath)


print("List of all authors whose works have been trained are:")
for author in train_authors:
    print(author)

testdatapath=os.path.join(py_path,"Test Data")

os.chdir(testdatapath)
test_cases=os.listdir()

corr=0
wrong=0
for case in test_cases:
    book_name=case.split(".txt")[0]
    f1=open(case,encoding="utf8")
    text=f1.read()
    token_words=word_tokenize(text)
    adverb_test=adverb_fraction(token_words)

    c1=normalize_counter(Counter(token_words))
    d1=dict(c1.most_common(len(c1)))

    bigram_test = nltk.bigrams(token_words)
    c2=normalize_counter(Counter(bigram_test))
    d2=dict(c2.most_common(len(c2)))

    trigram_test = nltk.trigrams(token_words)
    c3=normalize_counter(Counter(trigram_test))
    d3=dict(c3.most_common(len(c3)))

    error_author_dict={}
    bigram_error_author_dict={}
    trigram_error_author_dict={}


    adverb_error_author_dict={}

    for author in train_authors:
        #UnigramModel
        current_author_dict=author_dict[author]
        error=0
        max_uni_error=0
        for key in d1:
            max_uni_error=max_uni_error+abs(d1[key])
            if key in current_author_dict:
                error=error+abs(d1[key]-current_author_dict[key])  #Or try removing **2
            else:
                error=error+abs(d1[key])        
        error_author_dict[author]=error


        #BigramModel
        bigram_current_author_dict=bigram_author_dict[author]
        bigram_error=0
        max_bi_error=0
        for key in d2:
            max_bi_error=max_bi_error+abs(d2[key])
            if key in bigram_current_author_dict:
                bigram_error=bigram_error+abs(d2[key]-bigram_current_author_dict[key])
            else:
                bigram_error=bigram_error+abs(d2[key])
        bigram_error_author_dict[author]=bigram_error

        #TrigramModel
        trigram_current_author_dict=trigram_author_dict[author]
        trigram_error=0
        max_tri_error=0
        for key in d3:
            max_tri_error=max_tri_error+abs(d3[key])
            if key in trigram_current_author_dict:
                trigram_error=trigram_error+abs(d3[key]-trigram_current_author_dict[key])
            else:
                trigram_error=trigram_error+abs(d3[key])
        trigram_error_author_dict[author]=trigram_error

        adverb_error=abs(adverb_test-adverb_count_author[author])
        adverb_error_author_dict[author]=adverb_error*10000



    prob_author_dict={}
    prob_bigram_author_dict={}
    prob_trigram_author_dict={}

    master_prob_dict={}

    for key in error_author_dict:
        prob_author_dict[key]=(max_uni_error-error_author_dict[key])/max_uni_error

    for key in bigram_error_author_dict:
        prob_bigram_author_dict[key]=(max_bi_error-bigram_error_author_dict[key])/max_bi_error

    for key in trigram_error_author_dict:
        prob_trigram_author_dict[key]=(max_tri_error-trigram_error_author_dict[key])/max_tri_error


    for key in error_author_dict:
        master_prob_dict[key]=(0.99*prob_author_dict[key])+(0.005*prob_bigram_author_dict[key])+(0.005*prob_trigram_author_dict[key])*100


    key_min = min(error_author_dict.keys(), key=(lambda k: error_author_dict[k]))
    bigram_key_min = min(bigram_error_author_dict.keys(), key=(lambda k: bigram_error_author_dict[k]))
    trigram_key_min = min(trigram_error_author_dict.keys(), key=(lambda k: trigram_error_author_dict[k]))
    master_prob_max_key = max(master_prob_dict.keys(), key=(lambda k: master_prob_dict[k]))
    adv_key_min = min(adverb_error_author_dict.keys(), key=(lambda k: adverb_error_author_dict[k]))
#     print("*******")
#     print("The book '{}' is predicted to have been authored by '{}'".format(book_name,key_min))
#     print("Bigram model predicts {}".format(bigram_key_min))
#     print("Trigram model predicts {}".format(trigram_key_min))
    print("Interpolated model predicts {} was written by {}".format(book_name,master_prob_max_key))
    print("But adverb model predicts {}".format(adv_key_min))
#     print("*******")
    if adv_key_min==answer[case]:
        corr+=1
    else:
        wrong+=1
    f1.close()


print("Adverb prediction accuracy = {} %".format(100*corr/(corr+wrong)))