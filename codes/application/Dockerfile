# Use the specific Debian slim image
FROM debian:unstable-slim

# Set the working directory
WORKDIR /app

# Install Python, pip, and other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv build-essential libssl-dev libffi-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first
COPY requirements.txt /app/

# Create a Python virtual environment
RUN python3 -m venv /opt/venv

# Activate the virtual environment and install the dependencies
RUN . /opt/venv/bin/activate && pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code
COPY app.py /app/

# Expose the required port
EXPOSE 8501

# Set environment variable to prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Set the entrypoint to run the Streamlit application in the virtual environment
ENTRYPOINT ["/opt/venv/bin/streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
