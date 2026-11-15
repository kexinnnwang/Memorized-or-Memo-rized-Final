import time
from pathlib import Path
from beautiful_soup import scrape_w3schools_sql

HERE = Path(__file__).resolve().parent

def main():

    target_folder = HERE / "Documentation" / "SQL" / "SQL_original"

    # 2. Create a list of dictionaries containing the URLs and desired file names
    pages_to_scrape = [
           { "url": "https://www.w3schools.com/sql/sql_select.asp",
           "file_name": "SELECT.txt"},       
           {"url": "https://www.w3schools.com/sql/sql_min_max.asp",
               "file_name": "MIN().txt"},
           {"url": "https://www.w3schools.com/sql/sql_max.asp",
               "file_name": "MAX().txt"},     
           {"url": "https://www.w3schools.com/sql/sql_sum.asp",
               "file_name": "SUM().txt"},
           {"url": "https://www.w3schools.com/sql/sql_avg.asp",
               "file_name": "AVG().txt"},                   
    ]

    print(f"Scraping{len(pages_to_scrape)} pages...\n")
    print("-" * 40)
    
    for page in pages_to_scrape:
        current_url = page["url"]
        current_file = page["file_name"]                    
        
        try:        
            print(f"Scraping: {current_url}")   
            scrape_w3schools_sql(current_url, current_file, target_folder)
            print(f"Saving to: {current_file}")
        except Exception as e:
            print(f"Failed with Error: {e}")    
        print("-" * 40)                
        time.sleep(2)

    print("Completed!")

if __name__ == "__main__":
    main()