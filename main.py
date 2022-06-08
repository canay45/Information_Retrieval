##-----------------------------------------------------------------------##
##-------------------- CMPE 414 Information Retrieval -------------------##
## -------------------------Homework Assignment 1 -----------------------##
##Canay Kaplan 13975913008-----------------------------------------------##
##-----------------------------------------------------------------------##

import math
import docx
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from itertools import combinations

english_stopwords = stopwords.words('english')
all_words = []              ## this holds all documents' original terms
final_words = []            ## this holds terms which goes through pre-processing
TF_holder = {}              ## this holds each term and its TF value as a dictionary
DF_holder = {}              ## this holds each term and its DF value as a dictionary
IDF_holder = {}             ## this holds each term and its IDF value as a dictionary

def getText(filename):      ## function to get files
    doc = docx.Document(filename)
    fullText = ""
    for para in doc.paragraphs:
        fullText += para.text + '\t'
    return fullText


def textReader(filename,a):  ## function to read and apply pre-processing on documents which are docx format
    try:
        all_words.append(getText(filename))
        terms = punctuation_remover(all_words[a])
        lemmatized_terms = lemmatizer(terms)

        for w in lemmatized_terms:
            final_words.append(w)

        TF_computation(lemmatized_terms)
        DF_computation(lemmatized_terms)
        IDF_computation(lemmatized_terms,10)
        TF_IDF_computation(lemmatized_terms)
        print(f'TF_IDF values of doc{a}: {TF_IDF_computation(lemmatized_terms)}')
        return lemmatized_terms

    except IOError:
        print('There was an error opening the file!')
        return

def punctuation_remover(inp):           ## function for removing all punctuations
    tokenizer = RegexpTokenizer("\w+")  ## \w+ matches alphanumeric characters a-z,A-Z,0-9 and _
    tokens = tokenizer.tokenize(inp)
    new_word = " ".join(tokens)
    return new_word


def stopwords_remover(inp):             ## function for removing stopwords
    tokens = word_tokenize(inp)
    words_wo_stopwords = [a for a in tokens if a not in english_stopwords]
    removed_digits_words = ' '.join([i for i in words_wo_stopwords if not i.isdigit()])
    return removed_digits_words


def lemmatizer(text):                   ## function for lemmatizing texts
    lemmatizing_terms = WordNetLemmatizer()
    wo_stopwords_terms = stopwords_remover(text)
    word_list = wo_stopwords_terms.replace(u'\xa0', u' ').split()
    lemmatized_output = ' '.join([lemmatizing_terms.lemmatize(w.lower()) for w in word_list])
    return lemmatized_output.split()


def TF_computation(inp):                 ## function for calculating term-frequency(TF)
    term_dict = dict.fromkeys(inp, 0)

    for word in inp:
        term_dict[word] += 1
        frequency = sum(i for i in term_dict.values())

    for word in term_dict:
        result = (term_dict[word])/frequency
        print(f"Term: {word}, frequency: {term_dict[word]}, TF:{result}")

        if word in term_dict:
            key,value = word,result
            TF_holder[key] = value

    return inp


def DF_computation(w_inp):                  ## function for calculating document-frequency(DF)
    word_list = []
    for w in w_inp:
        if w not in word_list:
            word_list.append(w)
            if w in DF_holder:
                DF_holder[w] += 1
            else:
                DF_holder[w] = 1
    print(f'DF: {DF_holder}')

    return w_inp


def IDF_computation(w_inp,N):                  ## function for calculating IDF
    for w in w_inp:
        if w in DF_holder:
            IDF_holder[w] = math.log10( N / (DF_holder[w]))

    print(f'IDF: {IDF_holder}')
    return w_inp


def TF_IDF_computation(w_inp):                  ## function for calculating TF-IDF
    TF_IDF_holder = {}              ## this holds each term and its IDF value as a dictionary
    for word in w_inp:
        for w in DF_holder:
            if w in TF_holder:
                TF_IDF_holder[word] = TF_holder.get(word) * IDF_holder.get(word)
            else:
                TF_IDF_holder[word] = 0
    return TF_IDF_holder


def dot_prod(A, B):                 ## function for dot product operation which takes dictionary as a parameter
    return sum([a * b for a, b in zip(A, B)])


def cosine_similarity(a,b):         ## function for calculating cosine similarity
    dp_a = dot_prod(a,a)
    dp_b = dot_prod(b,b)

    sqrt_a = math.sqrt(dp_a)
    sqrt_b = math.sqrt(dp_b)

    result = dot_prod(a,b) / (sqrt_a * sqrt_b)
    return result



if __name__ == '__main__':

    similarity = {}
    calculated_similarity = {}
    read_doc = []

    index1 = 0
    index2 = 1

    for a in range(0,10):
        print(f'\n** document {a+1} is started **')
        read_doc.append(textReader(f'documents\doc{a}.docx',a))
    docs_comb = combinations(read_doc, 2)

    print()

    for e in list(docs_comb):
        inp1 = TF_IDF_computation(e[0])
        inp2 = TF_IDF_computation(e[1])

        string_version = (str(index1) + str(index2))
        similarity[string_version] = cosine_similarity(inp1.values(),inp2.values())
        index2 += 1
        if (index2 > 9):
            index1 += 1
            index2 = index1 + 1

    calculated_similarity = sorted(similarity.items(),key= lambda x: x[1], reverse=True)

    print()
    for found_ones in calculated_similarity:
        print(f'The cosine similarity of documents {found_ones[0][0]} and {found_ones[0][1]} is {found_ones[1]}')