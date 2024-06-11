import json
import os 

folder = ".\\data\\output\\DP_a2200_xml_ff2c81d8ad6655f915cbaa558ee7bf9e878730a8"

runtimes = []
for root,dir,files in os.walk(folder):
    for file in files:
        if 'results.json' == file:
            with open(os.path.join(root,file),"r") as _json:
                json_data = json.load(_json)
                runtimes.append(json_data["runtime"] / len(json_data["intermediate_results"]))
print(sum(runtimes)/len(runtimes))
