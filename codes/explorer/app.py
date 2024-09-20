import streamlit as st
from sqlalchemy import create_engine, inspect
import pandas as pd

st.set_page_config(page_title="Explorer", page_icon=":eyes:", layout="wide")
st.title("🔎 Logs Explorer")


# Database connection setup using SQLAlchemy
def get_db_connection():
    # Update with your own connection string
    db_url = f"postgresql+psycopg2://user:pass@postgres:5432/logging"
    engine = create_engine(db_url)
    return engine


# Function to retrieve all table names using SQLAlchemy's Inspector
def get_table_names(engine):
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    return tables


# Function to fetch all data from a specific table
def get_table_data(engine, table_name):
    query = f"SELECT * FROM {table_name};"
    with engine.connect() as connection:
        result = pd.read_sql(query, connection)
    return result


# Function to fetch all data from all tables
def get_all_tables_data(engine, table_names):
    tables_data = {}
    for table_name in table_names:
        table_data = get_table_data(engine, table_name)
        tables_data[table_name] = table_data
    return tables_data


# Create a database engine
engine = get_db_connection()

# Get all table names using SQLAlchemy's Inspector
table_names = get_table_names(engine)

# Add "View All" option to view data from all tables
if table_names:
    # Add option for viewing a specific table or all tables
    view_mode = st.radio(
        "Choose view mode", ["View all tables", "View a specific table"]
    )

    if view_mode == "View a specific table":
        selected_table = st.selectbox("Select a table to view", table_names)

        if selected_table:
            st.write(f"Data from `{selected_table}` table:")

            # Fetch and display table data
            table_data = get_table_data(engine, selected_table)

            if not table_data.empty:
                # Display the data in a table format
                st.dataframe(table_data)
            else:
                st.write("No data found in the table.")

    elif view_mode == "View all tables":
        # Fetch and display data from all tables
        all_tables_data = get_all_tables_data(engine, table_names)

        for table_name, table_data in all_tables_data.items():
            st.write(f"### Data from `{table_name}` table:")
            if not table_data.empty:
                st.dataframe(table_data)
            else:
                st.write(f"No data found in `{table_name}`.")
else:
    st.write("No tables found in the database.")
