import json
import os
import re
import sys
import sqlite3
import random

# Make the shared SQLite.Database class importable from this directory.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
from src.SQLite.sqlite_database import Database

# Same op tables as salesforce/WikiSQL lib/query.py
AGG_OPS = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_int(x):
    try:
        int(str(x))          
        return True
    except ValueError:
        return False

def get_table_name(table_dict):    
    title = table_dict.get("page_title") or table_dict.get("section_title") or table_dict["name"]
    
    return "_".join(title.split()).replace("__", "_")


def decode(sql, table_name, col_name):    
    agg = AGG_OPS[sql['agg']]
    return 'SELECT {}({}) FROM {}'.format(agg, col_name, table_name)

sql_text_map = {"SUM": "total", "AVG": "average", "MIN": "minimum", "MAX": "maximum"}
template_sql = "SELECT {operator}({value_name}) FROM {table_name}"
template_text_num = "What is the {operator} number of {value_name} of all {table_name}?"
template_text = "What is the {operator} {value_name} of all {table_name}?"


def clean(s):
    return re.sub(r"\W+", "_", s, flags=re.ASCII).strip("_")

def main():
    with open("train.tables.jsonl", "r", encoding="utf-8") as f:
        tables = [json.loads(line) for line in f if line.strip()]

    with open("train.jsonl", "r", encoding="utf-8") as f:
        train = [json.loads(line) for line in f if line.strip()]

    with open("../../Custom_Tables/table_template.txt", "r", encoding="utf-8") as f:
        table_template = f.read()     


    count = 0
    names = []
    data = []
    for table in tables:
        if "real" in table["types"] and "page_title" in list(table.keys()):
            name = get_table_name(table)
            if is_number(name[0]):
                continue
            
            if name not in names and re.fullmatch(r"\w+", name, re.ASCII) and len(table["page_title"].split()) > 3 and len(table["header"]) < 6:
                real = 0
                table_list =  table["rows"]    
                header_types = table["types"]

                keep_table = True
                for i in range(len(header_types)):
                    if header_types[i] == "real":
                        real += 1

                        for row in table_list:
                            if not is_int(row[i]):
                                keep_table = False
                                break
                if not keep_table:
                    continue

                for col_type in table["types"]:
                    if col_type == "real":
                        real += 1
                if real > 2 and real < 5:
                    names.append(name) 

                                                      
                    table_header = table["header"]
                    header_types = table["types"]

                    full_table = []
                    num_columns = []
                    header = []
                    for i in range(len(table_header)):
                        header.append(clean(table_header[i]))

                    for i in range(len(header_types)):
                        if header_types[i] == "text":
                            for row in table_list:
                                row[i] = clean(row[i])                  
                        else:
                            num_columns.append(header[i])
                    full_table.append(header)
                    full_table.extend(table_list)                    

                    for operator in sql_text_map.keys():
                        for value_name in num_columns:
                            question = template_text_num 

                            question = question.replace("{operator}", sql_text_map[operator])
                            question = question.replace("{value_name}", value_name)
                            question = question.replace("{table_name}", name)

                            sql = template_sql.replace("{operator}", operator)
                            sql = sql.replace("{value_name}", value_name)
                            sql = sql.replace("{table_name}", name)

                            print(full_table)
                            try:
                                mydb = Database({name: full_table})
                            except sqlite3.OperationalError:
                                continue 

                            gt = mydb.execute_query(sql)
                
                            if gt:
                                gt = [list(row) for row in gt]      
                            #print(gt)  

                            data.append({"question": question, "sql":sql, "table": {name: full_table}, "gt":list(gt)[0][0]}) 
                    curr_table_template = table_template.replace("{table_name}", name)
                    curr_table_template = curr_table_template.replace("{table}", str(full_table))
                    with open(f"../{name}.txt", "w") as f:        
                        f.write(curr_table_template) 
                
    random.shuffle(data)
    with open("../../Talkie_Train/simple_tables_sql_wiki.jsonl", "w") as f:
        for dat in data:
            f.write(json.dumps(dat) + "\n")        

if __name__ == '__main__':
    main()
