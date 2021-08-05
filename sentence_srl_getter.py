from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from pprint import pprint
import tqdm
from document_reader import *
from os import listdir
from os.path import isfile, join
import time
import json

predictor = Predictor.from_path("./structured-prediction-srl-bert.2020.12.15.tar.gz")
dataset = "HiEve"
dir_name = "../LEConstraints_EMNLP21/hievents_v2/processed/"

onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f)) and f[-4:] == "tsvx"]
onlyfiles.sort()
doc_id = -1
t0 = time.time()
for file_name in tqdm.tqdm(onlyfiles):
    doc_id += 1
    sent_id = []
    my_dict = []
    my_dict = tsvx_reader(dataset, dir_name, file_name)
    num_event = len(my_dict["event_dict"])
    for x in range(1, num_event+1):
        sent_id.append(my_dict["event_dict"][x]["sent_id"])
    for sid in sent_id:
        srl_result = predictor.predict(sentence = my_dict["sentences"][sid]['content'])
        my_dict["sentences"][sid]['srl'] = srl_result

    with open(file_name + '_sentences.json', 'w') as fp:
        json.dump(my_dict["sentences"], fp)
        print(file_name)