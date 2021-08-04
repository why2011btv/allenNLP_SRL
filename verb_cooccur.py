from os import listdir
from os.path import isfile, join 
import re
import csv
import pickle
import nltk
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
#nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

stop_words = set([])
def load_stop_words(filename):
    for line in open(filename):
        line = line.strip()
        stop_words.add(line)
    print ("Added", len(stop_words), "stop words.")
load_stop_words("/home1/w/why16gzl/KAIROS/event_abstraction/elmo/stop_words_en.txt")

mypath = '/shared/corpora-tmp/news_corpora/nyt/csv/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
csv.field_size_limit(100000000)

upper = "([A-Z])"
verb_set = set([])
verb_pair = {}
count_article = 0

for file_name in onlyfiles:
    print("file_name:", file_name)
    with open(mypath + file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for text in reader:
            text = text[-1].replace('\n','')
            #if ":" in text: text = text.replace(":","") # added by me
            text = re.sub(":" + upper,":. " + "\\1",text)
            text = re.sub("." + upper + upper + upper,". " + "\\1" + "\\2" + "\\3", text)
            #print(text)
            #if "To the Editor" in text: text = text.split("To the Editor")
            article = text
            if 1:
            #for article in text:
                #print(article)
                s_index_and_verb = set([])
                s_num = len(sent_tokenize(article))
                #for s_index, s in enumerate(split_into_sentences(text.lower())):
                for s_index, s in enumerate(sent_tokenize(article)):
                    #print(s)
                    pos_tags = nltk.pos_tag(word_tokenize(s.lower()))
                    for pos_tag in pos_tags:
                        verb = WordNetLemmatizer().lemmatize(pos_tag[0],'v')
                        if pos_tag[1] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"] and verb not in stop_words:
                            verb_set.add(verb)
                            s_index_and_verb.add((s_index, verb))
                #print("s_index_and_verb", s_index_and_verb, "\n")
                for siav_1 in s_index_and_verb:
                    for siav_2 in s_index_and_verb:
                        if siav_1[0] <= min(siav_2[0] + 3, s_num - 1) and siav_1[0] >= max(siav_2[0] - 3, 0) and siav_1[1] != siav_2[1]:
                            if (siav_1[1], siav_2[1]) not in verb_pair.keys():
                                verb_pair[(siav_1[1], siav_2[1])] = 1
                            else:
                                verb_pair[(siav_1[1], siav_2[1])] += 1

print(len(verb_pair))                                           
print(len(verb_set))

w = csv.writer(open("verb_pair.csv", "w"))
for key, val in verb_pair.items():
    w.writerow([key, val])
    
'''
f = open("verb_pair.pkl","wb")
pickle.dump(verb_pair,f)
f.close()
'''
f = open("verb_set.pkl","wb")
pickle.dump(verb_set,f)
f.close()
