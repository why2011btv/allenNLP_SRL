from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from sentence_splitter import split_into_sentences
from contextlib import ExitStack
import argparse
import json
import os
import csv
from os import listdir
from os.path import isfile, join

def simpleGeneratorFun(): 
    mypath = '/shared/corpora-tmp/news_corpora/nyt/csv/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    csv.field_size_limit(100000000)

    for file_name in onlyfiles:
        with open(mypath + file_name, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            counter = 0

            for row in reader:
                if counter > 0 and counter < 10:
                #if counter > 0:    
                    tmp_content = ""
                    for i in split_into_sentences(row[-1]):
                        tmp_content += i
                        tmp_content += "\n"
                    yield tmp_content
                counter += 1
            
class allenSRL: 
    def get_arguments():
        parser = argparse.ArgumentParser()
    
        parser.add_argument('--output-file', type=argparse.FileType('w'), help='path to output file')

        #parser.add_argument('input_file', type=argparse.FileType('r'), help='path to input file')

        parser.add_argument('--batch-size', type=int, default=1, help='The batch size to use for processing')



        args = parser.parse_args()

        return args

    def get_predictor():
       
        #return Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")
        print("loading predictor...")
        return Predictor.from_path("./srl-model-2018.05.25.tar.gz")


    def run(
        output_file,
        batch_size,
        print_to_console
        ):

        #predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz") 
        print("loading predictor...")
        predictor = Predictor.from_path("./srl-model-2018.05.25.tar.gz", cuda_device = 0) 
        print("predictor loaded!")
        def _run_predictor(batch_data):
                
                if len(batch_data) == 1:
                    result = predictor.predict_json(batch_data[0])
                    results = [result]
                else:
                    results = predictor.predict_batch_json(batch_data)
                return results

        srl_result = []
        for input_file in simpleGeneratorFun():
            batch_data = []
            batch_result = []
            for line in input_file:
                if not line.isspace():
                    line = {"sentence":line.strip()}
                    line = json.dumps(line)
                    json_data = predictor.load_line(line)
                    batch_data.append(json_data)
                    if len(batch_data) == batch_size:
                        res= _run_predictor(batch_data)
                        batch_result.append(res)
                        batch_data = []
            srl_result.append(batch_result)

        #if srl_result:
        #    res= _run_predictor(batch_data)
        return srl_result
       

    def main():
        
        args = allenSRL.get_arguments()
        predictor = allenSRL.get_predictor()
        output_file = None
        print_to_console = False

        with ExitStack() as stack:
        
                if args.output_file:
                    output_file = stack.enter_context(args.output_file)  

                if not args.output_file:
                    print_to_console = True

                result=allenSRL.run(output_file,args.batch_size,print_to_console)


                for output in result:
                    string_output = predictor.dump_line(output)
                    if print_to_console:
       
                        print("prediction: ", string_output)
                    if output_file:
                        output_file.write(string_output)
    
__end__ = '__end__'

if __name__ == '__main__':
    allenSRL.main()
    
 
