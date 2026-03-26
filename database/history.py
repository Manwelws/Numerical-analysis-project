import sqlite3


def init_db():
    con = sqlite3.connect("history.db")
    con.execute(""" Creat Table if not exists runs (
        id INTEGER PRIMARY KEY,
        method TEXT,
        equation TEXT,
        root REAL,
        iterations INTEGER,
        converged INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    con.commit()
    con.close()


def save_run(method, equation, root, iterations, converged):
    con = sqlite3.connect("history.db")
    con.execute(
        """
                insert into runs values(NULL,?,?,?,?,?,CURRENT_TIMESTAMP)""",
        (method, equation, root, iterations, converged),
    )
    con.commit()
    con.close()


def get_history():
    con = sqlite3.connect("history.db")
    rows = con.execute("Select * from runs ORDER BY timestamp DESC").fetchall()
    con.close
    return rows
