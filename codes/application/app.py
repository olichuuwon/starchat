"""
app.py
"""

import os
import jwt
import json
import uuid
import warnings
import requests
from datetime import datetime, timedelta

import streamlit as st

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Boolean,
    Text,
    TIMESTAMP,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama, VLLM
from langchain_community.utilities import SQLDatabase

from urllib3.exceptions import InsecureRequestWarning

# Suppress only the InsecureRequestWarning
warnings.simplefilter("ignore", InsecureRequestWarning)

# Load configurations from environment variables with default fallback values
CLIENT_ID = os.getenv("CLIENT_ID", "flask_client")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "fxAtVg6qe1eh78V4NurL3SeSNm2v8tUD")
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "https://keycloak.nebula.sl")
REALM = os.getenv("REALM", "text2sql")
REDIRECT_URI = os.getenv(
    "REDIRECT_URI", "https://application-route-starchat.apps.nebula.sl"
)

# Load model configurations from environment variables
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3:instruct")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Llama-3.1-8B-Instruct")

OLLAMA_MODEL_BASE_URL = os.getenv("OLLAMA_MODEL_BASE_URL", "http://model:11434")
VLLM_MODEL_BASE_URL = os.getenv(
    "VLLM_MODEL_BASE_URL", "https://llama-8b-route-sy-vllm.apps.nebula.sl/v2/models"
)

# Determine LLM provider (ollama or vllm)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()  # Default to "ollama"

# Load logging URL from environment variables
LOGGING_URL = os.getenv(
    "LOGGING_URL", "postgresql+psycopg2://user:pass@logging:5432/logging"
)
logging_engine = create_engine(LOGGING_URL)
logging_session = sessionmaker(bind=logging_engine)
base = declarative_base()


# Function to get current time in GMT+8
def get_gmt_plus_8_time():
    return datetime.now() + timedelta(hours=8)


class LogLogin(base):
    __tablename__ = "log_logins"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    is_successful = Column(Boolean, nullable=False)
    username = Column(Text, nullable=True)  # NULL allowed
    token = Column(Text, nullable=True)  # NULL allowed
    error_message = Column(Text, nullable=True)  # NULL allowed
    created_at = Column(
        TIMESTAMP, nullable=False, default=get_gmt_plus_8_time
    )  # Use datetime


class LogLLMInputsOutputs(base):
    __tablename__ = "log_llm_inputs_outputs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(Text, nullable=True)  # NULL allowed
    input = Column(Text, nullable=True)  # NULL allowed
    output = Column(Text, nullable=True)  # NULL allowed
    created_at = Column(
        TIMESTAMP, nullable=False, default=get_gmt_plus_8_time
    )  # Use datetime


class UserFeedback(base):
    __tablename__ = "user_feedbacks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(Text, nullable=True)  # NULL allowed
    scale = Column(Integer, nullable=False)
    feedback = Column(Text, nullable=False)
    created_at = Column(
        TIMESTAMP, nullable=False, default=get_gmt_plus_8_time
    )  # Use datetime


class UserSuggestion(base):
    __tablename__ = "user_sugguestions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(Text, nullable=True)  # NULL allowed
    suggestion = Column(Text, nullable=False)
    created_at = Column(
        TIMESTAMP, nullable=False, default=get_gmt_plus_8_time
    )  # Use datetime


def add_log_login(is_successful, username=None, token=None, error_message=None):
    session = logging_session()

    try:
        # Convert token to JSON string if it's a dictionary
        if isinstance(token, dict):
            token = json.dumps(token)

        new_log = LogLogin(
            is_successful=is_successful,
            username=username,
            token=token,
            error_message=error_message,
        )

        session.add(new_log)
        session.commit()
        print("LogLogin record added successfully.")

    except Exception as e:
        print(f"Error occurred while adding LogLogin: {e}")
        session.rollback()

    finally:
        session.close()


def add_llm_input_output(username, input_data=None, output_data=None):
    session = logging_session()

    try:
        new_entry = LogLLMInputsOutputs(
            username=username, input=input_data, output=output_data
        )

        session.add(new_entry)
        session.commit()
        print("LogLLMInputsOutputs record added successfully.")

    except Exception as e:
        print(f"Error occurred while adding LogLLMInputsOutputs: {e}")
        session.rollback()

    finally:
        session.close()


def add_user_feedback(username, scale, feedback):
    session = logging_session()

    try:
        new_feedback = UserFeedback(username=username, scale=scale, feedback=feedback)

        session.add(new_feedback)
        session.commit()
        print("UserFeedback record added successfully.")

    except Exception as e:
        print(f"Error occurred while adding UserFeedback: {e}")
        session.rollback()

    finally:
        session.close()


def add_user_suggestion(username, suggestion):
    session = logging_session()

    try:
        new_suggestion = UserSuggestion(username=username, suggestion=suggestion)

        session.add(new_suggestion)
        session.commit()
        print("UserFeedback record added successfully.")

    except Exception as e:
        print(f"Error occurred while adding UserFeedback: {e}")
        session.rollback()

    finally:
        session.close()


base.metadata.create_all(logging_engine)

st.set_page_config(page_title="Starchat", page_icon=":speech_balloon:", layout="wide")


def get_auth_code():
    """
    get auth code
    """
    auth_url = (
        f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/auth"
        f"?client_id={CLIENT_ID}&response_type=code"
        f"&redirect_uri={REDIRECT_URI}&scope=openid"
    )
    if "code" in st.query_params:
        code = st.query_params["code"]
        st.query_params.clear()
        return code
    else:
        st.markdown(
            f'<meta http-equiv="refresh" content="0; url={auth_url}">',
            unsafe_allow_html=True,
        )


def get_public_key():
    """
    get public key
    """
    well_known_url = f"{KEYCLOAK_URL}/realms/{REALM}/.well-known/openid-configuration"
    response = requests.get(well_known_url, verify=False)
    # st.write(response, response.json())
    cert_response = requests.get(response.json().get("jwks_uri"), verify=False)
    # st.write(cert_response, cert_response.json())
    cert_keys = cert_response.json().get("keys")
    # st.write(cert_keys)
    rsa_key = cert_keys[0]  # will rsa always be the first key?
    # st.write(rsa_key)
    return jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(rsa_key))


def get_access_token(auth_code):
    """
    get access token
    """
    try:

        token_url = f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/token"
        payload = {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": REDIRECT_URI,
        }
        response = requests.post(token_url, data=payload, verify=False)
        access_token = response.json().get("access_token")
        return access_token
    except Exception as e:
        st.error(f"get_access_token: {e}")
        return None


def get_decoded_token(access_token, public_key):
    """
    get decoded token
    """
    try:
        decoded_token = jwt.decode(
            access_token,
            key=public_key,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
        return decoded_token
    except Exception as e:
        st.error(f"get_decoded_token: {e}")
        return None


def authenticate():
    """
    authenticate
    """
    try:
        auth_code = get_auth_code()
        if auth_code is None:
            return None
        access_token = get_access_token(auth_code)
        if access_token is None:
            return None
        public_key = get_public_key()
        if public_key is None:
            return None
        decoded_token = get_decoded_token(access_token, public_key)
        if decoded_token is None:
            return None
        add_log_login(
            True,
            decoded_token["preferred_username"],
            token=decoded_token,
        )
        return decoded_token
    except Exception as e:
        add_log_login(
            False,
            error_message=e,
        )
        st.error(f"authenticate: {e}")
        return None


def get_chat_response(user_query, chat_history):
    """
    get chat response
    """
    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """

    # Create a prompt template from the provided template
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize the language model with the specified model name and base URL

    # Conditionally import the required library
    if LLM_PROVIDER == "ollama":
        # Initialize Ollama model
        llm = Ollama(
            model=OLLAMA_MODEL_NAME, base_url=OLLAMA_MODEL_BASE_URL, verbose=True
        )
    elif LLM_PROVIDER == "vllm":
        # Initialize VLLM model
        llm = VLLM(model=VLLM_MODEL_NAME, base_url=VLLM_MODEL_BASE_URL, verbose=True)
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")

    # Create a chain that processes the prompt and parses the output
    chain = prompt | llm | StrOutputParser()

    # Invoke the chain with the user's question and conversation history
    return chain.stream(
        {
            "chat_history": chat_history,
            "user_question": user_query,
        }
    )


def chat_mode_function():
    """
    chat mode function
    """
    st.title("🦛 Chat Mode")

    # Initialize chat history in session state if not already done
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a helpful assistant. How can I help you?"),
        ]

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    # Input for user query
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query.strip() != "":
        # Append the user's message to the chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        # Display the user's message
        with st.chat_message("Human"):
            st.markdown(user_query)

        # Generate and display the AI's response
        with st.chat_message("AI"):
            response = st.write_stream(
                get_chat_response(user_query, st.session_state.chat_history)
            )
        # Append the AI's response to the chat history
        st.session_state.chat_history.append(AIMessage(content=response))
        add_llm_input_output(
            st.session_state.jwt_token["preferred_username"],
            input_data=user_query,
            output_data=response,
        )


def database_mode_function():
    """
    database mode function
    """
    st.title("🐘 Database Mode")
    st.write("Coming soon...")
    if st.button("Submit Suggestion"):
        suggestion_form()  # Open the feedback form dialog


# Feedback dialog function
@st.dialog("Suggestion Form")
def suggestion_form():
    st.write("Please share your suggestions: ")
    suggestion = st.text_area(
        "How can large language models (LLMs) assist you in your daily tasks or improve your work experience? Share any specific use cases or challenges you'd like us to address!",
        max_chars=500,
    )

    if st.button("Submit"):
        if suggestion:  # Ensure feedback is not empty
            add_user_suggestion(
                st.session_state.jwt_token["preferred_username"], suggestion
            )
            st.success("Thank you for your sugguestions!")
        else:
            st.error("Please provide your suggestions before submitting.")


# Feedback dialog function
@st.dialog("Feedback Form")
def feedback_form():
    st.write("Please provide your feedback below:")
    scale = st.slider(
        "How would you rate this application? (0 - very unsatisfied, 10 - very satisfied)",
        0,
        10,
        5,
    )
    feedback = st.text_area("Let us know why:", max_chars=500)

    if st.button("Submit"):
        if feedback:  # Ensure feedback is not empty
            add_user_feedback(
                st.session_state.jwt_token["preferred_username"], scale, feedback
            )
            st.success("Thank you for your feedback!")
        else:
            st.error("Please provide your feedback before submitting.")


def main():
    """
    main
    """
    if "jwt_token" not in st.session_state:
        st.session_state.jwt_token = None

    if st.session_state.jwt_token is None:
        with st.spinner("Authenticating..."):
            st.session_state.jwt_token = authenticate()

    if st.session_state.jwt_token:
        st.sidebar.title("🌠 Starchat")
        # Main app logic
        modes = ["Chat Mode", "Database Mode"]
        if "page_selected" not in st.session_state:
            st.session_state.page_selected = modes[0]

        selected_mode = st.sidebar.radio("Select Mode", modes)
        st.session_state.page_selected = selected_mode
        if selected_mode == "Chat Mode":
            chat_mode_function()
        elif selected_mode == "Database Mode":
            database_mode_function()

        with st.sidebar:
            if st.button("Submit Feedback"):
                feedback_form()  # Open the feedback form dialog


if __name__ == "__main__":
    main()
