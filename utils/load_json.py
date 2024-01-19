import json
import os

def list_result_json(filepath):
    jsons = []
    for root,dir,files in os.walk(filepath):
        for file in files:
            if file == "results.json":
                jsons.append(os.path.join(root,file))
    return jsons
    
def load_result_json(filepath):
    _json = open(filepath)
    _json = json.load(_json)
    pdf_filename = _json["pdf_filename"]
    algorithm_results = _json["algorithm_results"]
    intermediate_results = _json["intermediate_results"]
    return pdf_filename,algorithm_results,intermediate_results

def load_results(filepath):
    pdfs = []
    algorithms = []
    intermediates =[]
    for path in list_result_json(filepath):
        pdf, algorithm, intermediate = load_result_json(path)
        pdfs.append(pdf)
        algorithms.append(algorithm)
        intermediates.append(intermediate)
    return pdfs,algorithms,intermediates
    