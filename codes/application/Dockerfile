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

# Set environment variables
ENV PYTHON_VERSION=3.12
ENV PIP_VERSION=23.1.2

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    libopenblas-dev \
    libomp-dev \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
# RUN python3 -m pip install --upgrade pip==${PIP_VERSION}

# Install PyTorch and torchvision with CUDA support
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 

# Expose the required port
EXPOSE 8501

# Set environment variable to prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Set the entrypoint to run the Streamlit application in the virtual environment
ENTRYPOINT ["/opt/venv/bin/streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
