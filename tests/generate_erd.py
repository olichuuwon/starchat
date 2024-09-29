import streamlit as st
from sqlalchemy import create_engine, inspect
from sqlalchemy.schema import Table, MetaData
import pydot
from graphviz import Source


"""
st.sidebar.success("Successfully connected to the database!")
                st.session_state.db = SQLDatabase.from_uri(
                    st.session_state.db_uri
                )  # Initialize database connection
"""


# Function to generate and display the ERD
def generate_erd(db_uri: str):
    engine = create_engine(db_uri)
    inspector = inspect(engine)
    metadata = MetaData()
    metadata.reflect(bind=engine)

    # Create a DOT file
    dot = pydot.Dot(graph_type="digraph")

    for table_name in inspector.get_table_names():
        table = Table(table_name, metadata, autoload_with=engine)
        # Create a node for each table
        table_node = pydot.Node(table_name, shape="box")
        dot.add_node(table_node)

        # Add relationships (foreign keys)
        for fk in table.foreign_keys:
            fk_table = fk.column.table.name
            fk_node = pydot.Node(fk_table, shape="box")
            dot.add_node(fk_node)
            edge = pydot.Edge(table_name, fk_table)
            dot.add_edge(edge)

    dot_content = dot.to_string()
    st.graphviz_chart(dot_content)


def main():
    # Streamlit UI to interactively display the ERD
    st.title("Interactive ERD Generator")

    db_uri = st.text_input(
        "Enter your database URI:",
        "postgresql+psycopg2://user:pass@postgres-text2sql.apps.nebula.sl:5432/chinook",
    )

    if st.button("Generate ERD"):
        if db_uri:
            generate_erd(db_uri)
        else:
            st.error("Please enter a valid database URI.")


if __name__ == "__main__":
    main()
