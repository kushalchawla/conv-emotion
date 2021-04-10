import random
import liwc
import nltk
from collections import Counter
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification, pipeline, T5ForConditionalGeneration, T5Tokenizer

class GoEmotionsRedditDataModel:
    
    def __init__(self):
        self.model = T5ForConditionalGeneration.from_pretrained("kiri-ai/t5-base-qa-summary-emotion")
        self.tokenizer = T5Tokenizer.from_pretrained("kiri-ai/t5-base-qa-summary-emotion")
        
    def predict(self, text):
        
        input_text = f"emotion: {text}"
        features = self.tokenizer([input_text], return_tensors='pt')
        tokens = self.model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'], max_length=64)
        out = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        print(out)
        
        return out

class EmoSemevalTwitterDataModel:
    
    def __init__(self):
        #load here
        self.tokenizer = AutoTokenizer.from_pretrained("lordtt13/emo-mobilebert")
        self.model = AutoModelForSequenceClassification.from_pretrained("lordtt13/emo-mobilebert")
        self.nlp_sentence_classif = pipeline('sentiment-analysis', model = self.model, tokenizer = self.tokenizer)
    
    def predict(self, text):
        
        out = self.nlp_sentence_classif(text)
        return out[0]['label']

class EmotionTwitterDataModel:
    
    def __init__(self):
        #load here
        self.tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
        self.model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    
    def predict(self, text):
        
        input_ids = self.tokenizer.encode(text + '</s>', return_tensors='pt')

        output = self.model.generate(input_ids=input_ids, max_length=2)
        
        dec = [self.tokenizer.decode(ids) for ids in output]
        
        label = dec[0].split()[-1]
        
        #assert label in ['joy', 'anger', 'sadness', 'love', 'fear', 'surprise'], dec
        return label

class LIWCModel:
    
    def __init__(self):
        #load here
        self.parse, self.category_names = liwc.load_token_parser('/home/ICT2000/chawla/Documents/work/CSCI662/project-EMNLP2020/main/resources/LIWC2015_English.dic')
        self.category_names = sorted(self.category_names)
        
    def predict(self, text):
        
        tokens = [w.lower() for w in nltk.word_tokenize(text)]
        feat_dict = dict(Counter(category for token in tokens for category in self.parse(token)))
    
        fname2val = {}
        
        for fname in self.category_names:
            fval = 0

            if(fname in feat_dict):
                fval = feat_dict[fname]
                
            fname2val[fname] = fval
        
        good_feats = ['negemo (Negative Emotions)', 'posemo (Positive Emotions)', 'anger (Anger)', 'anx (Anx)', 'sad (Sad)']
        good_vals = [fname2val[item] for item in good_feats]
        
        max_val = max(good_vals)
        
        if(max_val == 0):
            return 'Neutral'
        
        good_pairs = [(good_feats[i], good_vals[i]) for i in range(len(good_feats))]
        
        random.shuffle(good_pairs)
        
        for pair in good_pairs:
            if(pair[1] == max_val):
                #randomly picked max val
                return pair[0]
            
class EmoticonModel:
    def __init__(self):
        #load here
        self.em2name = {
            '🙂': 'Joy',
            '😮': 'Surprise',
            '☹️': 'Sadness', 
            '😡': 'Anger'
        }
    
    def predict(self, text):
        
        good_feats = list(self.em2name.keys())
        
        good_vals = [text.count(item) for item in good_feats]
        
        max_val = max(good_vals)
        
        if(max_val == 0):
            return 'Neutral'
        
        good_pairs = [(good_feats[i], good_vals[i]) for i in range(len(good_feats))]
        
        random.shuffle(good_pairs)
        
        for pair in good_pairs:
            if(pair[1] == max_val):
                #randomly picked max val
                return self.em2name[pair[0]]