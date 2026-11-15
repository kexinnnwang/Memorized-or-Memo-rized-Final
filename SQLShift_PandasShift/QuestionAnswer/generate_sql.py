import json
import random
import sys
from pathlib import Path

# Make the shared SQLite.Database class importable from this directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.SQLite.sqlite_database import *

sql_text_map = {"SUM": "total", "AVG": "average", "MIN": "minimum", "MAX": "maximum"}
template_sql = "SELECT {operator}({value_name}) FROM {table_name}"
template_text_num = "What is the {operator} number of {value_name} of all {table_name}?"
template_text = "What is the {operator} {value_name} of all {table_name}?"


employees = [["employee_id", "department", "salary", "age", "years_experience", "projects"], [1, "Sales", 50, 28, 3, 5], [2, "Engineering", 65, 34, 8, 12], [3, "Marketing", 55, 41, 15, 20], [4, "Engineering", 70, 29, 5, 7], [5, "Sales", 48, 25, 2, 3]]
orders = [["order_id", "customer_id", "product_id", "quantity", "price"], [1, 101, 2, 1, 900], [2, 102, 1, 3, 135], [3, 103, 3, 10, 20], [4, 104, 4, 2, 300]]
products = [["product_id", "category", "price", "stock", "rating", "weight"], [1, "Furniture", 45, 120, 4, 3500], [2, "Electronics", 900, 30, 5, 1800], [3, "Stationery", 2, 500, 3, 10], [4, "Furniture", 150, 40, 4, 12000], [5, "Electronics", 600, 25, 4, 180], [6, "Stationery", 3, 300, 5, 20]]
movies = [["movie_id", "genre", "year", "rating", "duration", "revenue"], [1, "Action", 2019, 7, 128, 350], [2, "Comedy", 2021, 6, 95, 80], [3, "Drama", 2018, 8, 142, 120], [4, "Action", 2022, 7, 135, 410], [5, "Horror", 2020, 5, 88, 60], [6, "Drama", 2023, 9, 150, 200]]
restaurants = [["restaurant_id", "cuisine", "rating", "seats", "avg_price"], [1, "Italian", 4, 60, 25], [2, "Mexican", 3, 40, 15], [3, "Japanese", 5, 30, 35], [4, "Italian", 4, 80, 28], [5, "Indian", 4, 50, 20]]
books = [["book_id", "author_id", "pages", "year", "price"], [1, 201, 320, 2015, 15], [2, 202, 450, 2018, 22], [3, 201, 280, 2020, 18], [4, 203, 510, 2012, 25], [5, 204, 190, 2021, 12]]
athletes = [["athlete_id", "sport", "age", "height", "wins", "losses"], [1, "Tennis", 27, 185, 45, 12], [2, "Boxing", 31, 178, 30, 5], [3, "Swimming", 24, 190, 60, 8], [4, "Tennis", 22, 175, 20, 15], [5, "Boxing", 29, 182, 38, 10], [6, "Swimming", 26, 188, 52, 6]]
students = [["student_id", "grade", "gpa", "age", "attendance", "credits"], [1, 9, 35, 14, 95, 24], [2, 10, 38, 15, 92, 30], [3, 11, 29, 16, 88, 36], [4, 12, 40, 17, 97, 42], [5, 9, 33, 14, 90, 24]]


employees_int = ["salary", "age", "years_experience", "projects"]
orders_int = ["quantity", "price"]
products_int = ["price", "stock", "rating", "weight"]
students_int = ["gpa", "age", "attendance", "credits"]
movies_int = ["rating", "duration", "revenue"]
books_int = ["pages", "price"]
restaurants_int = ["rating", "seats", "avg_price"]
atheletes_int = ["age", "height", "wins", "losses"]

tables = {
    "employees": {"data": employees, "int_columns": employees_int},
    "orders": {"data": orders, "int_columns": orders_int},
    "products": {"data": products, "int_columns": products_int},
    "athletes": {"data": athletes, "int_columns": atheletes_int},
    "movies": {"data": movies, "int_columns": movies_int},
    "books": {"data": books, "int_columns": books_int},
    "restaurants": {"data": restaurants, "int_columns": restaurants_int},
    "students": {"data": students, "int_columns": students_int}
}

def generate_data():
    data = []

    for table_name, table_info in tables.items():
        mydb = Database({table_name: table_info["data"]})
        for operator in sql_text_map.keys():
            for value_name in table_info["int_columns"]:
                if value_name in ["years_experience", "projects", "pages", "seats", "wins", "losses", "credits"]:
                    question = template_text_num                        
                else:
                    question = template_text

                question = question.replace("{operator}", sql_text_map[operator])
                question = question.replace("{value_name}", value_name)
                question = question.replace("{table_name}", table_name)

                sql = template_sql.replace("{operator}", operator)
                sql = sql.replace("{value_name}", value_name)
                sql = sql.replace("{table_name}", table_name)

                gt = mydb.execute_query(sql)
                
                if gt:
                    gt = [list(row) for row in gt]        

                data.append({"question": question, "sql":sql, "table": {table_name: table_info["data"]}, "gt":list(gt)[0][0]})
    random.shuffle(data)
    with open(Path(__file__).resolve().parent / "simple_tables_sql.jsonl", "w") as f:
        for dat in data:
            f.write(json.dumps(dat) + "\n")        

if __name__ == "__main__":
    generate_data()
             