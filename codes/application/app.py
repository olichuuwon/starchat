"""
app.py
"""

import os
import jwt
import json
import uuid
import warnings
import requests
import http.cookies
from datetime import datetime, timedelta
from time import sleep

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
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from urllib3.exceptions import InsecureRequestWarning

# Suppress only the InsecureRequestWarning
warnings.simplefilter("ignore", InsecureRequestWarning)

# KEYCLOAK AUTH
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL")
REALM = os.getenv("REALM")
REDIRECT_URI = os.getenv("REDIRECT_URI")

# OLLAMA MODEL
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME")
OLLAMA_MODEL_BASE_URL = os.getenv("OLLAMA_MODEL_BASE_URL")

# VLLM MODEL
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME")
VLLM_FULL_MODEL = os.getenv("VLLM_FULL_MODEL")

# Determine LLM provider (ollama or vllm)
LLM_PROVIDER = os.getenv("LLM_PROVIDER")

# Determine AUTH provider (keycloak-api or oauth-proxy)
AUTH_PROVIDER = os.getenv("AUTH_PROVIDER")

# Cookie key if oauth-proxy
PROXY_JWT_KEY = os.getenv("PROXY_JWT_KEY")
WEB_LINK = os.getenv("WEB_LINK")

# Where logs go
LOGGING_ENDPOINT = os.getenv("LOGGING_ENDPOINT")
# Logging engine
logging_engine = create_engine(LOGGING_ENDPOINT)
logging_session = sessionmaker(bind=logging_engine)
base = declarative_base()

# Prompt engineering
template = """
<|begin_of_text|>

<|start_header_id|>assistant<|end_header_id|>
You are a direct and concise assistant. Answer the user’s latest question clearly and succinctly.
After responding to the first question, wait for more User input.
<|eot_id|>

{chat_history}

<|start_header_id|>user<|end_header_id|>
{user_question}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""


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


def keycloak_api():
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


def oauth_proxy():
    """
    piggyback authentication
    """
    session = requests.Session()
    cookie = session.cookies.get_dict()
    if PROXY_JWT_KEY not in cookie:
        print("Awaiting cookie...")
        cookie[PROXY_JWT_KEY] = {"preferred_username": "cookie_identity"}
    else:
        print(cookie[PROXY_JWT_KEY])
    return cookie[PROXY_JWT_KEY]


def get_chat_response(user_query, chat_history):
    """
    Get chat response by sending a request to the Triton VLLM API.
    """
    # Prepare an explicit template to minimize extra explanations

    # Format the prompt
    prompt = ChatPromptTemplate.from_template(template)
    prompt_text = prompt.format(chat_history=chat_history, user_question=user_query)

    # st.write(prompt_text)

    if LLM_PROVIDER == "vllm":
        headers = {"Content-Type": "application/json"}
        payload = {
            "text_input": prompt_text,
            "parameters": {"stream": False, "temperature": 0, "max_tokens": 1000},
        }

        response = requests.post(
            VLLM_FULL_MODEL, json=payload, headers=headers, verify=False
        )

        if response.status_code != 200:
            raise Exception(f"Failed to get response from Triton API: {response.text}")

        # Assuming the response is in JSON format and extracting the relevant part
        data = response.json()
        assistant_response = data.get("text_output", "")
        procesesed_response = assistant_response.split(
            "<|start_header_id|>assistant<|end_header_id|>"
        )[-1]

        return procesesed_response  # Return the cleaned and concise response

    elif LLM_PROVIDER == "ollama":
        # Initialize Ollama model
        llm = Ollama(
            model=OLLAMA_MODEL_NAME, base_url=OLLAMA_MODEL_BASE_URL, verbose=True
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke(
            {
                "chat_history": chat_history,
                "user_question": user_query,
            }
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")


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
            response = get_chat_response(user_query, st.session_state.chat_history)
            typewriter_effect(response)

        # Append the AI's response to the chat history
        st.session_state.chat_history.append(AIMessage(content=response))
        add_llm_input_output(
            st.session_state.jwt_token["preferred_username"],
            input_data=user_query,
            output_data=response,
        )


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


def typewriter_effect(text, speed=0.001):
    """
    Function to display text in a typewriter effect.
    :param text: The full text to display.
    :param speed: Speed of typing (in seconds).
    """
    placeholder = st.empty()  # Create a placeholder to display the text
    typed_text = ""

    # Add characters one by one to the typed_text and update the placeholder
    for char in text:
        typed_text += char
        placeholder.markdown(typed_text)  # Update the text displayed in the placeholder
        sleep(speed)  # Introduce delay to simulate typing


def main():
    """
    main
    """

    if "jwt_token" not in st.session_state:
        st.session_state.jwt_token = {}

    if st.session_state.jwt_token == {} and AUTH_PROVIDER == "keycloak-api":
        st.session_state.jwt_token = keycloak_api()
    elif st.session_state.jwt_token == {} and AUTH_PROVIDER == "oauth-proxy":
        st.session_state.jwt_token = oauth_proxy()
    elif st.session_state.jwt_token == {} and AUTH_PROVIDER == "none":
        st.session_state.jwt_token["preferred_username"] = "unknown-user"

    if st.session_state.jwt_token:
        st.sidebar.title("🌠 Starchat")
        st.sidebar.header(f"Hello, {st.session_state.jwt_token['preferred_username']}!")
        chat_mode_function()
        with st.sidebar:
            st.caption("How was your experience using our platform?")
            if st.button("Share Feedback"):
                feedback_form()  # Open the feedback form dialog
            st.caption(
                "Are there any tools or features that could help streamline your workflows?"
            )
            if st.button("Share Suggestion"):
                suggestion_form()  # Open the suggestion form dialog
            st.caption("Would you like to restart the conversation and clear history?")
            if st.button("Restart Chat History"):
                st.session_state.chat_history = [
                    AIMessage(
                        content="Hello, I am a helpful assistant. How can I help you?"
                    ),
                ]
                st.rerun()


if __name__ == "__main__":
    main()
