import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os 
import json
import re

def scrape_w3schools_sql(url, file_name, folder):    
    os.makedirs(folder, exist_ok=True)    
    output_filepath = os.path.join(folder, f"{file_name}")    

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}    

    syntax_in_h3 = ['COUNT()', 'GROUP BY', 'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN']



    try:        
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        soup = BeautifulSoup(response.text, 'html.parser')
        with open(output_filepath, 'w', encoding='utf-8') as out_file:

            main_content = soup.find(id='main')    
            if main_content:               
                # Extract Operator
                title = main_content.find('h1')                
                if title:
                    operator = title.text.strip().replace('SQL', '').replace('Statement', '')   
                    operator = operator.replace("Keyword", "")            
                    operator = operator.replace("Clause", "")            
                    operator = operator.replace("Operator", "")            
                    operator = operator.replace("Function", "")            
                    operator = " ".join(operator.split())
                    print(f"OPERATOR: {operator}\n", file=out_file)
                else:
                    print("Could not find the operator title.\n")
                
                # Extract Description
                paragraphs = main_content.find_all('p', recursive=False)
                first_h2 = main_content.find('h2')
                if first_h2:        
                    print("DESCRIPTION:", file=out_file)                                
                    for sibling in first_h2.find_next_siblings():
                        if sibling.name == 'p':            
                            text = sibling.text.strip().replace('\n', ' ')
                            if text:
                                text = " ".join(text.split())                                       
                                print(f"{text}", file=out_file)
                        elif sibling.name in ['ul', 'ol']:
                            for li in sibling.find_all('li'):
                                text = li.text.strip().replace('\n', ' ')
                                if text:
                                    text = " ".join(text.split())
                                    print(f"- {text}", file=out_file)
                        elif sibling.name in ['h2', 'h3', 'div']:                    
                            break
                    print("\n", file=out_file)
                else:
                    print("No h2 tag found for description.\n", file=out_file)
                
                # Extract Syntax  
                if operator in syntax_in_h3:
                    syntax_tag =  main_content.find(lambda tag: tag.name == 'h3' and f'{operator} Syntax' in tag.text)                    
                    if syntax_tag:            
                        example_div = syntax_tag.find_next_sibling('div', class_='w3-example')      
                        #example_div = syntax_tag.find_next_sibling('div', class_='w3-example')            
                        if example_div:                                                
                            code_block = example_div.find('code')
                            if not code_block:
                                code_block = example_div.find('div', class_='w3-code')      
                        # if example_div:                    
                        #     code_block = example_div.find('code')       
                             
                            if code_block:       
                                print("SYNTAX:", file=out_file)               
                                syntax_text = code_block.get_text(separator=' ').replace('\n ', '\n').replace(' ;', ';').strip()    
                                syntax_text = " ".join(syntax_text.split())                                       
                                print(syntax_text, file=out_file)                        
                                print('\n', file=out_file)                        
                            else:
                                print("Could not find the <code> tag inside the div.", file=out_file)
                        else:
                            print("Could not find the w3-example div after the h2.", file=out_file)
                    else:
                        print(f"Could not find an h3 with {operator} Syntax.\n", file=out_file)
                else:
                    syntax_tag =  main_content.find(lambda tag: tag.name == 'h2' and (f'Syntax' in tag.text))                    
                    if syntax_tag:            
                        example_div = syntax_tag.find_next_sibling('div', class_='w3-example')            
                        if example_div:                    
                            code_block = example_div.find('code')            
                            if code_block:       
                                print("SYNTAX:", file=out_file)               
                                syntax_text = code_block.get_text(separator=' ').replace('\n ', '\n').replace(' ;', ';').strip()    
                                syntax_text = " ".join(syntax_text.split())                                       
                                print(syntax_text, file=out_file)                        
                                print('\n', file=out_file)                        
                            else:
                                print("Could not find the <code> tag inside the div.", file=out_file)
                        else:
                            print("Could not find the w3-example div after the h2.", file=out_file)
                    else:
                        print(f"Could not find an h2 with {operator} Syntax.\n", file=out_file)
            
                # Example Database
                database_tag = soup.find(lambda tag: tag.name == 'h2' and 'Demo Database' in tag.text)
                if database_tag:
                    print("EXAMPLE WITH SAMPLE DATABASE AND SAMPLE CODE:", file=out_file)
                    
                    all_databases = {}
                    
                    for p in database_tag.find_all_next('p'):
                        table_name = None
                                                
                        strong = p.find('strong')
                        if strong and 'table' in p.text.lower():
                            table_name = strong.text.strip()
                        
                        if not table_name:
                            match = re.search(r'["""]([^"""]+)["""]\s*table', p.text)
                            if match:
                                table_name = match.group(1)
                        
                        if not table_name:
                            continue
                        
                        table = p.find_next('table', class_='ws-table-all')
                        if not table:
                            continue
                        
                        # Extract headers
                        header_row = table.find('tr')
                        if not header_row:
                            continue
                        headers = [th.text.strip() for th in header_row.find_all('th')]
                        
                        # Build list-of-lists
                        table_data = [headers]
                        for tr in table.find_all('tr')[1:]:
                            cells = tr.find_all('td')
                            if cells and len(cells) == len(headers):
                                row = [cell.text.strip() for cell in cells]
                                table_data.append(row)
                        
                        all_databases[table_name] = table_data
                    
                    if all_databases:
                        for name, data in all_databases.items():
                            print(f"{name} = {data}", file=out_file)
                    else:
                        print("Could not find any tables under the heading.", file=out_file)
                else:
                    print("Could not find an h2 with Demo Database.", file=out_file)
                                   
                # Example Code
                demo_h2 = soup.find(lambda tag: tag.name == 'h2' and 'Demo Database' in tag.text)
                if demo_h2:                                
                    subsequent_h2s = demo_h2.find_all_next('h2')
                    extracted_examples = []

                    for h2 in subsequent_h2s:
                        example_name = h2.text.strip()
                        description_paragraphs = []
                        sql_code = None
                        found_example_box = False
                                            
                        for sibling in h2.find_next_siblings():                
                            if sibling.name == 'h2':
                                break                                             
                            if sibling.name == 'p':
                                description_text = sibling.text.strip().replace('\n', ' ')
                                description_text = " ".join(description_text.split())                                                                                            
                                description_paragraphs.append(description_text)                                                    
                            if sibling.name == 'div' and 'w3-example' in sibling.get('class', []):
                                found_example_box = True                                                    
                                code_div = sibling.find('div', class_='w3-code')
                                if code_div:
                                    sql_code = code_div.get_text(separator=' ').replace('\n ', '\n').strip()   
                                    sql_code = " ".join(sql_code.split())                                                                                            
                                break 
                                
                        if found_example_box:
                            extracted_examples.append({'Name': example_name, 'Description': ' '.join(description_paragraphs), 'Code': sql_code})                        
                    
                    for i, ex in enumerate(extracted_examples, 1):
                        print(f"--- Sample Code {i} ---", file=out_file)                        
                        print(f"Sample Code Description: \n{ex['Name']}. {ex['Description']}", file=out_file)
                        print(f"Sample SQL Code:\n{ex['Code']}", file=out_file)                       
                else:
                    print("Could not find the 'Demo Database' heading.", file=out_file)                              

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while trying to fetch the page: {e}")