import string 
import re 
from nltk import SnowballStemmer
from nltk.stem import PorterStemmer, LancasterStemmer

def choose_stemmer(text,stem="sbs"):
        # Create stemmer
    if stem == 'sbs':
        stemmer = SnowballStemmer(language='english')
    elif stem == 'p':
        stemmer = PorterStemmer()
    elif stem == 'l':
        stemmer = LancasterStemmer()    
    else:
        raise Exception("stem has to be 'sbs' for Snowball 'p' for Porter or 'l' for Lancaster")
    # Stemming
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

def stemming_review(data):
    data['reviews'] = data['reviews'].apply(choose_stemmer)
    data.reviews.head()
    return data

def positive_negative_review(data):

    data = data[data.stars!=3]
    # 4/5 Yıldız --> POZİTİF, 1/2 Yıldız --> NEGATİF
    data['sentiment'] = data['stars'].apply(lambda x: (x>=4 and 'Positive') or 'Negative')
    data = data[['sentiment','stars','reviews']]
    #data.head()
    return data



def preprocess_text(data):

    # Yorumların içerisindeki <br> ifadelerinden kurtulma
    data['reviews'] = data['reviews'].str.replace("<br />","")
    # Yorumlar içerisinde geçen sayısal değer barındıran kelimeleri kaldırma
    alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
    # Noktalama işaretlerini kaldırma ve tüm kelimeleri küçük harfe dönüştürme
    punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower()) 

    data['reviews'] = data.reviews.map(alphanumeric).map(punc_lower)

    return data
