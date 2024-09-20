"""
python -m streamlit run c:/Users/Jesly/Workspace/Text2SQL/text-to-sql/codes/streamlit/streamlit_auth_test.py
"""

import streamlit as st
import jwt  # Import the JWT library
from jwt import DecodeError, ExpiredSignatureError
import requests
import json

CLIENT_ID = "flask_client"
CLIENT_SECRET = "fxAtVg6qe1eh78V4NurL3SeSNm2v8tUD"
KEYCLOAK_URL = "https://keycloak.nebula.sl"
REALM = "text2sql"
REDIRECT_URI = "http://localhost:8501"  # Streamlit default port

# State management for Streamlit
if "auth_code" not in st.session_state:
    st.session_state.auth_code = None

if "access_token" not in st.session_state:
    st.session_state.access_token = None


# Fetch the public key from Keycloak for JWT verification
def get_keycloak_public_key():
    well_known_url = f"{KEYCLOAK_URL}/realms/{REALM}/.well-known/openid-configuration"
    response = requests.get(well_known_url, verify=False)

    if response.status_code != 200:
        st.error(f"Failed to fetch well-known config: {response.text}")
        st.stop()

    jwks_uri = response.json().get("jwks_uri")
    jwks_response = requests.get(jwks_uri, verify=False)

    if jwks_response.status_code != 200:
        st.error(f"Failed to fetch JWKS keys: {jwks_response.text}")
        st.stop()

    jwks_keys = jwks_response.json().get("keys")

    # Assuming only one key is used for signing
    if not jwks_keys or len(jwks_keys) == 0:
        st.error("No JWKS keys found")
        st.stop()

    rsa_key = jwks_keys[0]  # Directly using the first key

    # Convert JWKS to PEM format key
    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(rsa_key))

    return public_key


def main():
    # Streamlit UI
    st.title("Keycloak Auth App")

    if st.session_state.auth_code is None:
        st.write("Please click the button below to begin the authentication process.")
        if st.button("Authenticate with Keycloak"):
            auth_url = (
                f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/auth"
                f"?client_id={CLIENT_ID}&response_type=code"
                f"&redirect_uri={REDIRECT_URI}&scope=openid"
            )
            st.markdown(f"[Authenticate with Keycloak]({auth_url})")

    else:
        st.write(f"Authorization code received: {st.session_state.auth_code}")

        # Exchange the authorization code for an access token
        token_url = f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/token"
        payload = {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "authorization_code",
            "code": st.session_state.auth_code,
            "redirect_uri": REDIRECT_URI,
        }

        response = requests.post(token_url, data=payload, verify=False)

        # Log the response for debugging
        st.write(f"Token exchange response status: {response.status_code}")

        if response.status_code == 200:
            token_info = response.json()
            st.session_state.access_token = token_info.get("access_token")
            st.write(f"Access Token: {st.session_state.access_token}")

            # Decode the JWT using the public key
            try:
                public_key = get_keycloak_public_key()
                decoded_token = jwt.decode(
                    st.session_state.access_token,
                    key=public_key,
                    algorithms=["RS256"],
                    options={"verify_aud": False},  # Disable audience verification
                )
                st.write("Decoded Token:", decoded_token)
                pretty_decoded_token = json.dumps(decoded_token, indent=4)
                st.code(pretty_decoded_token)

            except ExpiredSignatureError:
                st.error("Token has expired")
            except DecodeError:
                st.error("Token decode error")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            error_message = response.json().get("error_description", "Unknown error")
            st.error(
                f"Failed to obtain access token: {response.status_code} {error_message}"
            )

    # Handle authorization code after redirect
    params = st.query_params
    if "code" in params and st.session_state.auth_code is None:
        st.session_state.auth_code = params["code"]
        st.rerun()


if __name__ == "__main__":
    main()
