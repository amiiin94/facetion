import os
import pandas as pd
from datetime import datetime

LOG_PATH = 'attendance_log.csv'
CLASS_PATH = 'classes.csv'

def ensure_csv_files():
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=["Name", "Class", "Date", "Time"]).to_csv(LOG_PATH, index=False)
    if not os.path.exists(CLASS_PATH):
        pd.DataFrame(columns=["Class Name", "Date"]).to_csv(CLASS_PATH, index=False)

def add_class(name, date):
    df = pd.read_csv(CLASS_PATH)
    df = pd.concat([df, pd.DataFrame([[name, date]], columns=["Class Name", "Date"])], ignore_index=True)
    df.to_csv(CLASS_PATH, index=False)

def load_classes():
    return pd.read_csv(CLASS_PATH)

def log_attendance(name, class_name):
    now = datetime.now()
    log = pd.read_csv(LOG_PATH)
    new_row = pd.DataFrame([[name, class_name, now.date(), now.strftime("%H:%M:%S")]], columns=log.columns)
    log = pd.concat([log, new_row], ignore_index=True)
    log.to_csv(LOG_PATH, index=False)

def view_attendance_log():
    return pd.read_csv(LOG_PATH)

def get_registered_students(database):
    """Return the list of student names from the database"""
    # Simply return the names list without modification
    return database["names"]

def get_student_display_names(database):
    """Return a list of student names with NIMs if available for display purposes"""
    if "nims" in database:
        # Return formatted strings with name and NIM where available
        return [f"{name} ({nim})" if nim else name 
                for name, nim in zip(database["names"], database["nims"])]
    return database["names"]

def delete_student_by_name(database, name):
    if name in database["names"]:
        idx = database["names"].index(name)
        database["names"].pop(idx)
        database["embeddings"].pop(idx)
        if "uuids" in database:
            database["uuids"].pop(idx)
        if "nims" in database:
            database["nims"].pop(idx)
    return database

def delete_student_by_uuid(database, uuid_to_delete):
    """Delete a student from the database by UUID."""
    if "uuids" in database and uuid_to_delete in database["uuids"]:
        index = database["uuids"].index(uuid_to_delete)
        name = database["names"][index]  # Get name for reference/success message
        
        # Remove entry from all lists
        database["names"].pop(index)
        database["embeddings"].pop(index)
        database["uuids"].pop(index)
        if "nims" in database:
            database["nims"].pop(index)
        
        return True, name
    return False, ""
