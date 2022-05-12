import re
import nltk
nltk.data.path.append('/nltk_data/')
# from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import TweetTokenizer
import contractions
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import sys


class TweetCleaner():

    def __init__(self) -> None:
        
        self.stopwords = self._load_stopwords()
        self.tweetTokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        self.lem = WordNetLemmatizer()
        self.valid_words = self._load_words()
        self.non_english_tokens = self._load_non_english_tokens()

    def _load_stopwords(self):

        with open('stop.txt', 'r') as stop:
            words = stop.read().strip().split('\n')
        return words


    def _cleanTweetsRegex(self, x):
        #remove urls
        pattern_link = r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])'
        x = re.sub(pattern_link, '', x)
        
        #remove punctuations
        pattern_punct = r'[.\?!,\[\]{}_"]'
        x = re.sub(pattern_punct, '', x)
        
        #remove special chars
        pattern_special = r'[#$%^&\*|\\/<>¦]'
        x = re.sub(pattern_special, '', x)
        
        #remove numbers
        pattern_nums = r'[0-9]+'
        x = re.sub(pattern_nums, '', x)
        
        return x


    def _tokenizeTweets(self, tweet, tokenizer=None):

        if tokenizer is None:
            tokenizer = self.tweetTokenizer

        #remove isolated chars which make up emoji
        single_char_emoji = [':',';','(',')','-']
        return ' '.join([word for word in tokenizer.tokenize(tweet) if word not in single_char_emoji])

    

    def _nltk_tag_to_wordnet_tag(self,nltk_tag):

        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
        
    def _lemmatize_token(self, word):
        lst = []
        lst.append(word)
        wordtpl = pos_tag(lst)
        tag = self._nltk_tag_to_wordnet_tag(wordtpl[0][1])
        if(tag!=None):
            return self.lem.lemmatize(word,tag)
        else:
            return word

    def _normalize_text(self, x):
        stop = self.stopwords
        x = self._tokenizeTweets(x)
        x = contractions.expand_contractions(x)
        return ' '.join([self._lemmatize_token(word) for word in x.split() if word not in stop])

    def _load_words(self):
        with open('words.txt') as word_file:
            valid_words = set([word.lower() for word in word_file.read().split()])
        return valid_words

    def _load_non_english_tokens(self):
        
        with open('nonenglishtokens.txt') as net:
            words = set([word.lower() for word in net.read().split()])
        return words

    def clean(self,tweet):

        clean_reg = self._cleanTweetsRegex(tweet)
        tokenized = self._normalize_text(clean_reg)
        x = re.sub(r'@', '', tokenized)
        final_tweet = ' '.join([word for word in x.split() if word not in self.non_english_tokens])
        tweet_arr = np.array([final_tweet])
        return tweet_arr




def recall_m( y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m( y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))



class TensorflowTweet():

    def __init__(self, path_to_model) -> None:
        self.max_length = 50
        self.padding_type = 'post'
        self.trunc_type = 'post'
        self.tc = TweetCleaner()
        self.tokenizer = Tokenizer(oov_token='<OOV>')
        self.model = load_model(path_to_model, custom_objects={"f1_m":f1_m})

    def _prepare(self, tweet):

        print(f"inside prepare before fit, data is {tweet}", file=sys.stderr)
        self.tokenizer.fit_on_texts(tweet)
         
        text_sequence = self.tokenizer.texts_to_sequences(tweet)
        print(f"inside prepare before pad, data is {text_sequence}", file=sys.stderr)
        padded_sequence = pad_sequences(text_sequence,maxlen=self.max_length, padding=self.padding_type, truncating=self.trunc_type)
        print(f"inside prepare after pad, data is {padded_sequence}", file=sys.stderr)

        padded_sequence = np.expand_dims(padded_sequence, -1)
        print(f"inside prepare after pad, data is {padded_sequence.shape}", file=sys.stderr)

        return padded_sequence

    
    def predict(self,tweet):
        print(f"inside predict, data is {tweet}", file=sys.stderr)
        cleaned = self.tc.clean(tweet=tweet)
        print(f"inside predict, after clean data is {cleaned}", file=sys.stderr)
        data = self._prepare(cleaned)
        print(f"inside predict, after prepare data is {data}", file=sys.stderr)
        prediction = self.model.predict(data)
        print(f"inside predict, after prepare prediction is {prediction.shape}", file=sys.stderr)
        result = dict()
        result['probability'] = str(prediction[0,0])
        if np.round(prediction[0,0]) == 0:
            result['decision'] = "not Hate Speech"
        else:
            result['decision'] = "Hate Speech"
        return result


        




