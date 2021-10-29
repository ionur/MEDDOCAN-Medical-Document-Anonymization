# -*- coding: utf-8 -*-
"""milestone-4-ext1.ipynb
"""

from nltk.corpus import conll2002
from nltk.corpus import cess_esp as cess

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
import pickle
# !pip3 install wordfreq
from collections import OrderedDict 
from wordfreq import zipf_frequency

# !pip3 install pandas
import pandas as pd
import string
import datetime

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#Load the names from .csv file to python
names_male = pd.read_csv('../male_names.csv',)
names_female = pd.read_csv('../female_names.csv')

#Create the male dict
male_dict={}

for index, row in names_male.iterrows():
    _key = row['name']
    _val1 = row['mean_age']
    _val2 = row['frequency']
    male_dict[_key] = (_val1,_val2)
    
    
#Create the female dict
female_dict={}

for index, row in names_female.iterrows():
    _key = row['name']
    _val1 = row['mean_age']
    _val2 = row['frequency']
    female_dict[_key] = (_val1,_val2)

#Locations Dictionary
#Taken from https://en.wikipedia.org/wiki/List_of_municipalities_of_Spain
locations=['Madrid','Barcelona','Valencia','Seville','Zaragoza','Málaga','Murcia','Palma',
        'Las Palmas de Gran Canaria','Bilbao','Alicante','Córdoba','Valladolid',
        'Vigo','Gijón','L Hospitalet de Llobregat','A Coruña','Vitoria-Gasteiz',
        'Granada','Elche','Oviedo','Badalona','Cartagena','Terrassa','Jerez de la Frontera']

        #'Sabadell','Santa Cruz de Tenerife','Móstoles','Alcalá de Henares','Pamplona','Fuenlabrada',
        #'Almería','Leganés','Donostia-San Sebastián','Burgos','Santander',
        #'Castellón de la Plana','Getafe','Albacete',
        #'Alcorcón','Logroño','San Cristóbal de La Laguna','Badajoz','Salamanca','Huelva','Lleida',
        #'Marbella','Tarragona','León','Cádiz','Tineo','Baza','Alcántara','Don Benito',
        #'Piedrabuena','Alhambra','Sabiñánigo','Montoro','Torrelavega','Guadalajara','Palencia',
        #'Vic–Manlleu','Ourense']

locations_set = set()

for location in locations:
  if location.upper() not in locations_set:
    locations_set.add(location.upper())

def isApostrophePresent(word):
    if "'" in word:
        return True
    return False

  
def isDashPresent(word):
    if "-" in word:
        return True
    return False

def oneDigit(word):
    num = 0
    for l in word:
      if l.isdigit():
        num +=1
    if num == 1:
      return True
    else:
      return False

def twoDigits(word):
    num = 0
    for l in word:
      if l.isdigit():
        num +=1
    if num == 2:
      return True
    else:
      return False
    
def threeDigits(word):
    num = 0
    for l in word:
      if l.isdigit():
        num +=1
    if num == 3:
      return True
    else:
      return False
    
def fiveDigits(word):
    num = 0
    for l in word:
      if l.isdigit():
        num +=1
    if num == 5:
      return True
    else:
      return False
    
def sixDigits(word):
    num = 0
    for l in word:
      if l.isdigit():
        num +=1
    if num == 6:
      return True
    else:
      return False
    
def sevenDigits(word):
    num = 0
    for l in word:
      if l.isdigit():
        num +=1
    if num == 7:
      return True
    else:
      return False
    
def nineDigits(word):
    num = 0
    for l in word:
      if l.isdigit():
        num +=1
    if num == 9:
      return True
    else:
      return False
    
def fax(word):
    if "fax" in word.lower():
      return True
    else:
      return False

def isAge(word):
  if "edad" in word.lower() or "años" in word.lower():
    return True
  
  return False

    
def hasPunctuation(word):
  #This might return a lot of false positives
  
  for letter in word:
    if( letter  in string.punctuation):
      return True
    
  return False


def isRoman(word):
  #This might return a lot of false positives
  romans = ['I','V','M','L','X','D','C']
  
  for letter in word:
    if( letter not in romans):
      return False
    
  return True

def isDigit(word):
  for letter in word:
    if not letter.isdigit():
      return False
    
  return True

def maleFeatures(word):
   if word in male_dict:
    return male_dict[word]
   else:
    return (0,0)
    

def femaleFeatures(word):
   if word in female_dict:
    return female_dict[word]
   else:
    return (0,0)
  
def isLocation(word):
  if word.upper() in locations_set:
    return True
  else:
    return False

def getfeats(word, postag, o):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)
    _male = maleFeatures(word)
    _female = femaleFeatures(word)
    _isMale = False
    _isFemale = False
    
    if( _male[0] != 0 or _male[1] != 0) :
        _isMale = True
        
    if( _female[0] != 0 or _female[1] != 0) :
        _isFemale = True
    
    features = [
        (o + 'word', word), #0
        
        (o + 'word.len', len(word) ), #1
        (o + 'oneDigit', oneDigit(word)), #2
        (o + 'twoDigits', twoDigits(word)), #3
        (o + 'threeDigits', threeDigits(word)), #4
        (o + 'fiveDigits', fiveDigits(word)), #5
        (o + 'sixDigits', sixDigits(word)), #6
        (o + 'sevenDigits', sevenDigits(word)), #7
        (o + 'nineDigits', nineDigits(word)), #8
        (o + 'word.isupper', any(letter.isupper() for letter in word)), #9
        (o + 'hasPunctuation', hasPunctuation(word)), #10
        (o + 'isRoman', isRoman(word)), #11
        (o + 'age', isAge(word)), #12
        (o + 'isupper', word.isupper()), #13
        (o + 'islower', word.islower()), #14
        (o + 'isApostrophePresent', isApostrophePresent(word)), #15
        (o + 'isDashPresent', isDashPresent(word)), #16
        (o + 'fax', fax(word)), #17
   
       
        (o + 'word.wordfreq', zipf_frequency(word, 'es') ), #23
        (o + 'word_count_en', zipf_frequency(word, 'en')) , #24
        (o + 'isMale', _isMale ), #16
        (o + 'maleAvgAge', _male[0] ), #17
        (o + 'maleAvgFreq', _male[1]), #18
        (o + 'isFemale', not _isMale ), #19
        (o + 'femaleAvgAge', _female[0] ), #20
        (o + 'femaleAvgFreq', _female[1] ), #21
        (o + 'isLocation', isLocation(word)), #22
        (o + 'postag', postag), #23
        
    ]
    return features

count = 0

def word2features(sent, i):
    global count
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token
    featlist = [('bias', 1.0)]
    features.extend(featlist)
  
    #print('Processed :',count,' sentences')
    count = count+1
    for o in [-1,0,1,2]:
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            postag = sent[i+o][3]
#             postag = "unknown"
            featlist = getfeats(word, postag, o)
            features.extend(featlist)
        elif i+o<0:
            featlist = [('BOS', 1)]
            features.extend(featlist)
        else:
            featlist = [('EOS', 1)]
            features.extend(featlist)    
    return dict(features)

#create an .ann format for predictions
def createAnnFormat(preprocess_dict, y_pred, pathRead, pathOut):   
  
  
    j = 0
    for docId, v in preprocess_dict.items():
        #print('Generating .ann for ',docId)
#         #Get the y_pred
#         X = preprocess_dict[docId]        
        #make predictions and store the results with their tags
        txt_start_end = ""
        tmp_tags=[]
        curr_tag = ""
        
        for sent in v:
          for item in sent:
            word = item[0]
            word_start= item[2]
            word_end = word_start + len(word) - 1
            pred = y_pred[j]   

            #if prediction is a beginning, add the tag to the tag list
            if pred[0:2] == "B-":
              if txt_start_end != "":
                tmp_tags.append((txt_start_end,curr_tag))
                txt_start_end= ""
              curr_tag= pred[2:]
              txt_start_end +=""+str(word_start)+"-"+str(word_end)+","
            #if it is a contuniation keep adding 
            elif pred[0:2] == "I-":
              txt_start_end +=""+str(word_start)+"-"+str(word_end)+","
            j += 1
        if txt_start_end != "":
          tmp_tags.append((txt_start_end,curr_tag))
          txt_start_end= ""  
        #we might be missing some chars in between values, so parse the data and 
        #get sentence matching that                        
        with open(pathRead+docId+".txt", "r") as f:                                                   
          complete_doc = f.read()       

        #now for all the text, get their exact match and create tags
        tags = []
        t_count = 1                        
        for i, i_tag in tmp_tags:
          splits = i.split(",")[:-1]
          beginning_split = splits[0].split("-")                 
          real_start = int(beginning_split[0])                      
          if len(splits) ==1:                  
            real_end = int(beginning_split[1])+1
          else:
            end_split = splits[len(splits)-1].split("-")
            real_end = int(end_split[1])+1
          text_appearing = complete_doc[real_start:real_end].split("\n")[0]
          tags.append("T"+str(t_count)+"\t"+i_tag+" "+str(real_start)+" "+str(real_end)+"\t"+ text_appearing)
          t_count += 1
        #now output all these
        with open(pathOut+docId+".ann", "w") as out:
          for i in tags:
            out.write(i+"\n")


def createDictionary(preprocess_dict):
  type_dictionary = {}
  
  for k,v in preprocess_dict.items():
    for sentence in v:
      last_index=0
      found = False
      
      for i in range(len(sentence)):
        #go through the words to see when you hit a ':'
        word = sentence[i][0]
        
        if word == ':':  
          field_name = ""
          if not found:       
            for j in sentence[0:i]:
              field_name +=j[0]+" "
            field_name= field_name.strip()
          else:
            j = i-1
            punctuations = ['.',',','-',':',")"]
            fields =[]
            while(j>=0):
              if sentence[j][1] == 'O' and sentence[j][0] not in punctuations:
                fields.append(sentence[j][0])
                j-=1
              else:
                j=-1
            for x in fields[::-1]:
              field_name += x+" "
            field_name = field_name.strip()
          found = True
          if i+1 < len(sentence): 
            next_tag = sentence[i+1][1]
            if next_tag != 'O':
              type_dictionary[field_name] = next_tag[2:]
        
#         #if there are multiple fields in a sentence
#         if found:
#           if sentence[i][1] == 'O':
#             last_index=i
  return type_dictionary

def regexRules(preprocess_dict, type_dict):
  y_pred = []
  for k,v in preprocess_dict.items():
    for sentence in v:
      #check how many ':' there are
      indices =[]
      
      for i in range(len(sentence)):
        #go through the words to see when you hit a ':'
        word = sentence[i][0]
        if word ==":":
          indices.append(i)
          
      if len(indices) == 1:
        index = indices[0]
        field_name = ""
        for i in range(index):
          field_name += sentence[i][0]+" "
          y_pred.append('O')
        field_type = type_dict[field_name.strip()]
        y_pred.append('O')
        
        for i in range(index+1,len(sentence)):
          if i==index+1:
            y_pred.append('B-'+field_type)
          elif sentence[i][0] != '.':
            y_pred.append('I-'+field_type)
          else:
            y_pred.append('O')
            
      #go backwards from indices
      elif len(indices) > 1:
        y_backwards= []
        end = len(sentence)
        
        for i in range(len(indices)):
          j=indices[len(indices)-i-1]-1
          curr_type = ""
          counter = 0
          while(j>=0 and curr_type not in type_dict.keys()):
            curr_type =sentence[j][0]+" "+curr_type
            j -= 1
            counter +=1
            if counter ==5:
              break
          
          if curr_type in type_dict.keys():
            field_type = type_dict[curr_type]

            for x in range(indices[len(indices)-i-1]+1,end):
              if x==indices[len(indices)-i-1]+1:
                y_backwards.append('B-'+field_type)
              elif sentence[x][0] != '.':
                y_backwards.append('I-'+field_type)
              else:
                y_backwards.append('O')
            for x in range(j,indices[len(indices)-i-1]+1):
              y_backwards.append('O')
            end=j
          else:
            for x in range(j,end):
              y_backwards.append('O')
            end = j
          
      #just O
      else:
        for i in range(len(sentence)):
          y_pred.append('O')
  return y_pred

if __name__ == "__main__":
  
    # Load the training data
    file = open('../train_word_ner_startidx_dict.pickle','rb')
    train_dict = pickle.load(file)

    train_dict = OrderedDict(train_dict)
    
    #use the training data to create a dictionary from field names to types
    #Milestone 2 enhancement cont.
    types_dictionary  = createDictionary(train_dict)
    
    file.close()
    
    train_sents = []
    
    for k, v in train_dict.items():
      train_sents.extend(v)

    print("train_sents len= ", len(train_sents))

    file = open('../dev_word_ner_startidx_dict.pickle','rb')
    dev_dict = pickle.load(file)
    

    dev_dict = OrderedDict(dev_dict)
    file.close()
    
    dev_sents = []
    for k, v in dev_dict.items():
      dev_sents.extend(v)
    
    print("dev_sents len= ", len(dev_sents))
      
    file = open('../test_word_ner_startidx_dict.pickle','rb')
    test_dict = pickle.load(file)
    
  
    test_dict = OrderedDict(test_dict)
    file.close()
    
    test_sents = []
    for k, v in test_dict.items():
      test_sents.extend(v)
    
    print("test_sents len= ", len(test_sents))
    
    train_feats = []
    train_labels = []

    print('Started preparing the features',datetime.datetime.now())
    for sent in train_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i)
            train_feats.append(feats)
            train_labels.append(sent[i][1])

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)
    print('Finished preparing the features',datetime.datetime.now())

    #make sure these models are ordered from best to worst performance
    models = [LinearSVC(C=1),RandomForestClassifier(n_estimators=25),LogisticRegression(),DecisionTreeClassifier()]
    #give weights
    model_weights=[1.2,0.8,0.6,0.4]
    
    
    #Dev Data
    dev_feats = []
    dev_labels = []

    for sent in dev_sents:
      for i in range(len(sent)):
          feats = word2features(sent,i)
          dev_feats.append(feats)
          dev_labels.append(sent[i][1])
    X_dev = vectorizer.transform(dev_feats)
    
    #Test Data
    test_feats = []
    test_labels = []

    # switch to test_sents for your final results
    for sent in test_sents:
      for i in range(len(sent)):
          feats = word2features(sent,i)
          test_feats.append(feats)
          test_labels.append(sent[i][1])

    X_test = vectorizer.transform(test_feats)
    
    
    all_predictions = {'train':{},'dev':{},'test':{}}
    majority_predictions = {'train':[],'dev':[],'test':[]}

       
    for i,model in enumerate(models):
      print('Starting to fit model ',i, datetime.datetime.now())
      model.fit(X_train, train_labels)
      print('Finished the model ', i, datetime.datetime.now())
      all_predictions['train'][i] = model.predict(X_train)
      all_predictions['dev'][i] = model.predict(X_dev)
      all_predictions['test'][i] = model.predict(X_test)
   
    
    
    #majority rule method:
    for dataset_name, val in all_predictions.items():
      for i in range(len(val[0])):  
        majority_guess = {}
        for j in range(len(models)):
          guess = val[j][i]
          if guess in majority_guess.keys():
            majority_guess[guess] += model_weights[j]
          else:
            majority_guess[guess] = model_weights[j]
        winner = sorted(majority_guess.items(), key=lambda kv: kv[1],reverse=True)[0][0]
        majority_predictions[dataset_name].append(winner)

    train_pathRead = "../../data/train/system/"
    train_pathOut = "../../data/train/system/"
    
    #create an .ann format for predictions
    createAnnFormat(train_dict, majority_predictions['train'],train_pathRead, train_pathOut)
    
    dev_pathRead = "../../data/dev/system/"
    dev_pathOut = "../../data/dev/system/"

    createAnnFormat(dev_dict, majority_predictions['dev'],dev_pathRead, dev_pathOut)
    
    test_pathRead = "../../data/test/system/"
    test_pathOut = "../../output/test/system/"
    
    createAnnFormat(test_dict, majority_predictions['test'],test_pathRead, test_pathOut)  
    
    # evaluate_model("Majority Rule")
