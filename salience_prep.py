from document_reader import *
from os import listdir
from os.path import isfile, join
import networkx as nx
import json
from pprint import pprint

def _verb_position_(tags):
    count = -1
    for i in tags:
        count += 1
        if i == 'B-V':
            return count

def find_arg(srl_description, arg):
    """
    /shared/why16gzl/logic_driven/NAACL_2021/comet-commonsense/why.py
    """
    arg_start = srl_description.find(arg)
    if arg_start == -1:
        return ""
    else:
        arg = srl_description[(arg_start + len(arg) + 2):]
        arg_end = arg.find("]")
        arg = arg[0:arg_end]
        return arg
    
def obtain_v_plus_o(srl_result, mention, token_id):
    """
    /shared/why16gzl/logic_driven/NAACL_2021/comet-commonsense/why.py
    """
    #srl_result = srl_predictor.predict(sentence = sentence)
    #token_list = srl_result["words"]
    arg0 = ""
    arg1 = ""
    for verb_dict in srl_result["verbs"]:
        if verb_dict["verb"] == mention:
            verb_position = _verb_position_(verb_dict['tags'])
            if verb_position > token_id - 3 and verb_position < token_id + 3:
                arg0 = find_arg(verb_dict["description"], "ARG0")
                arg1 = find_arg(verb_dict["description"], "ARG1")
                return_value_0 = arg0 + ' [V: ' + mention + "] " + arg1   # [V: was]
                return_value_0 = return_value_0.strip()
                return return_value_0
    return '[NomEvent: ' + mention + ']'

dir_name = "../LEConstraints_EMNLP21/hievents_v2/processed/"
onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f)) and f[-4:] == "tsvx"]
onlyfiles.sort()
#onlyfiles = ["article-10901.tsvx"]

root_sum = 0
subevent_sum = 0
event_complex = {}
ec_id = -1
for file_name in tqdm.tqdm(onlyfiles):
    with open('sentences_json/' + file_name + '_sentences.json') as f:
        sentences = json.load(f)
    my_dict = tsvx_reader('HiEve', dir_name, file_name)
    num_event = len(my_dict["event_dict"])
    event_info = {}
    for x in range(1, num_event+1):
        sid = my_dict['event_dict'][x]["sent_id"]
        words = sentences[sid]['srl']['words']
        token_id = my_dict['event_dict'][x]["token_id"]
        mention = my_dict['event_dict'][x]["mention"]
        event_info[x] = obtain_v_plus_o(sentences[sid]['srl'], mention, token_id)
        
    '''
    /shared/why16gzl/Repositories/LEConstraints_EMNLP21/eventseg_getter.py
    '''
    G = nx.Graph()
    DG = nx.DiGraph()

    for (event_id1, event_id2) in my_dict["relation_dict"].keys():
        if my_dict["relation_dict"][(event_id1, event_id2)]['relation'] == 0:
            G.add_edge(int(event_id1), int(event_id2))
            DG.add_edge(int(event_id1), int(event_id2))
        elif my_dict["relation_dict"][(event_id1, event_id2)]['relation'] == 1:
            G.add_edge(int(event_id2), int(event_id1))
            DG.add_edge(int(event_id2), int(event_id1))
        else:
            do_nothing = 1
            
    roots = [n for n,d in DG.in_degree() if d==0] 
    root_sum += len(roots)
    for root in roots:
        ec_id += 1
        subevents = []
        for connected_component in list(nx.connected_components(G)): 
            if root in connected_component:
                connected_component.remove(root)
                for event_id in list(connected_component):
                    if event_info[event_id] != event_info[root]:
                        subevents.append(event_info[event_id])
        subevent_sum += len(subevents)   
        event_complex[ec_id] = {event_info[root]: subevents}      
        
#pprint(event_complex)        
print("avg roots per documents", float(root_sum) / len(onlyfiles))
print("subevents / event complex", float(subevent_sum) / root_sum)

with open('event.complex.json', 'w') as fout:
    json.dump(event_complex, fout)