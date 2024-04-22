import json
import os

def list_result_json(filepath):
    '''
    This function is reponsible for listing the all "results" json file in the folder including subdirectories
    
    param filepath: the filepath to a folder contains json files
    
    Return:
    a list contains all "results" json files in the folder
    '''
    jsons = []
    for root,dir,files in os.walk(filepath):
        for file in files:
            if file == "results.json":
                jsons.append(os.path.join(root,file))
    return jsons
    
def load_result_json(filepath):
    '''
    This function is to read a json file, get and separate 'pdf_filename', 'algorithm_results' and 'intermediate_results' from the json file
    
    param filepath: the filepath to a json file that has key 'pdf_filename', 'algorithm_results' and 'intermediate_results'
    
    Return:
    string: pdf file name
    list: a list of algorithms results of the page
    list: a list of intermediate results of the page 
    '''
    _json = open(filepath)
    _json = json.load(_json)
    pdf_filename = _json["pdf_filename"]
    algorithm_results = _json["algorithm_results"]
    intermediate_results = _json["intermediate_results"]
    return pdf_filename,algorithm_results,intermediate_results

def load_results(filepath):
    '''
    The function to load json files, and separate pdf name, algorithm results, and intermediate results in a folder
    
    param filepath: the folder that contains the json files
    
    Return:
    list: a list of pdf name
    list: a list of lists of algorithm results
    list: a list of lists of intermediate results
    '''
    pdfs = []
    algorithms = []
    intermediates =[]
    for path in list_result_json(filepath):
        pdf, algorithm, intermediate = load_result_json(path)
        pdfs.append(pdf)
        algorithms.append(algorithm)
        intermediates.append(intermediate)
    return pdfs,algorithms,intermediates
    