import sqlite3
from contextlib import contextmanager




class Database:
    def __init__(self, initial_values: dict) -> None:
        self.con = sqlite3.connect(":memory:")
        self.con.row_factory = sqlite3.Row
        cur = self.con.cursor()
        for name, values in initial_values.items():
            columns = values[0]            
            placeholders = ", ".join(["?"] * len(columns))
            updated_name = name.replace("-", "_")
            cur.execute(f"CREATE TABLE {updated_name}({", ".join(columns)})")
            cur.executemany(
                f"INSERT INTO {updated_name} VALUES({placeholders})", values[1:]
            )
        self.con.commit()

    def execute_query(self, query: str) -> list:        
        updated_query = query.replace("article-author", "article_author")    
        updated_query = updated_query.replace("citing-cited", "citing_cited")        
        # return self.con.execute(updated_query).fetchall()
        try:
            return self.con.execute(updated_query).fetchall()
        #except sqlite3.OperationalError as e:
        except sqlite3.Error as e:
            #print(query)
            #print(f"Error executing query: {e}")
            return False
