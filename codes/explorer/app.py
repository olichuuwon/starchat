import streamlit as st
from sqlalchemy import create_engine, inspect
import pandas as pd


# Database connection setup using SQLAlchemy
def get_db_connection():
    # Update with your own connection string
    db_url = "postgresql://your_user:your_password@your_host/your_database"
    engine = create_engine(db_url)
    return engine


# Function to retrieve all table names using SQLAlchemy's Inspector
def get_table_names(engine):
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    return tables


# Function to fetch all data from a table
def get_table_data(engine, table_name):
    query = f"SELECT * FROM {table_name};"
    with engine.connect() as connection:
        result = pd.read_sql(query, connection)
    return result


# Streamlit app interface
st.title("PostgreSQL Table Viewer with SQLAlchemy")

# Create a database engine
engine = get_db_connection()

# Get all table names using SQLAlchemy's Inspector
table_names = get_table_names(engine)

if table_names:
    selected_table = st.selectbox("Select a table to view", table_names)

    if selected_table:
        st.write(f"Data from `{selected_table}` table:")

        # Fetch and display table data using Pandas
        table_data = get_table_data(engine, selected_table)

        if not table_data.empty:
            # Display the data in a table format
            st.dataframe(table_data)
        else:
            st.write("No data found in the table.")
else:
    st.write("No tables found in the database.")
