import sqlite3
from datetime import datetime

def init_db(db_path="attendance.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Check if tables exist and get their structure
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = [table[0] for table in c.fetchall()]
    
    # Create or update users table
    if 'users' not in existing_tables:
        # Create new users table
        c.execute("""
        CREATE TABLE users (
            id INTEGER,
            user_id TEXT PRIMARY KEY,
            name TEXT,
            created_at TEXT
        )""")
    else:
        # Check if we need to add the created_at column
        c.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in c.fetchall()]
        if 'created_at' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
    
    # Create or update face_samples table
    if 'face_samples' not in existing_tables:
        c.execute("""
        CREATE TABLE IF NOT EXISTS face_samples (
            user_id TEXT PRIMARY KEY, 
            face_samples BLOB,
            num_samples INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )""")
    
    # Update logs table or create it
    if 'logs' not in existing_tables:
        c.execute("""
        CREATE TABLE logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            name TEXT,
            timestamp TEXT
        )""")
    else:
        # Check if we need to add the name column
        c.execute("PRAGMA table_info(logs)")
        columns = [column[1] for column in c.fetchall()]
        if 'name' not in columns:
            c.execute("ALTER TABLE logs ADD COLUMN name TEXT")
    
    conn.commit()
    return conn

# Add new user
def add_user(conn, user_id, name):
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute("INSERT OR REPLACE INTO users (user_id, name, created_at) VALUES (?,?,?)",
              (user_id, name, now))
    conn.commit()

# Add face samples for a user
def add_face_samples(conn, user_id, face_samples):
    c = conn.cursor()
    num_samples = len(face_samples)
    c.execute("INSERT OR REPLACE INTO face_samples (user_id, face_samples, num_samples) VALUES (?,?,?)",
              (user_id, face_samples, num_samples))
    conn.commit()

# Get face samples for a user
def get_face_samples(conn, user_id):
    c = conn.cursor()
    result = c.execute("SELECT face_samples FROM face_samples WHERE user_id=?", (user_id,)).fetchone()
    if result:
        return result[0]
    return None

# Delete a user and their face samples
def delete_user(conn, user_id):
    c = conn.cursor()
    c.execute("DELETE FROM face_samples WHERE user_id=?", (user_id,))
    c.execute("DELETE FROM users WHERE user_id=?", (user_id,))
    conn.commit()

# Record check-in
def add_log(conn, user_id, name=None):
    c = conn.cursor()
    now = datetime.now().isoformat()
    
    # If name not provided, look it up
    if name is None:
        result = c.execute("SELECT name FROM users WHERE user_id=?", (user_id,)).fetchone()
        if result:
            name = result[0]
        else:
            name = "Unknown"
            
    c.execute("INSERT INTO logs (user_id, name, timestamp) VALUES (?,?,?)", 
              (user_id, name, now))
    conn.commit()

# Fetch all users
def fetch_users(conn):
    c = conn.cursor()
    users = c.execute("""
        SELECT u.user_id, u.name, u.created_at, COALESCE(f.num_samples, 0) as face_count 
        FROM users u 
        LEFT JOIN face_samples f ON u.user_id = f.user_id
        ORDER BY u.name
    """).fetchall()
    return users

# Fetch logs with user names
def fetch_logs(conn, limit=50):
    c = conn.cursor()
    return c.execute("""
        SELECT l.user_id, COALESCE(l.name, 'Unknown') as name, l.timestamp 
        FROM logs l
        ORDER BY l.timestamp DESC
        LIMIT ?
    """, (limit,)).fetchall()