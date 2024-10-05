# Get alphine python image
FROM python:3.12-alpine 

# Setup env
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV ACCEPT_EULA=Y

RUN apk update && apk add --no-cache \
    gcc \
    g++ \
    curl \
    unixodbc-dev \
    bash \
    libffi-dev \
    openssl-dev \
    cargo \
    musl-dev \
    postgresql-dev \
    cmake \
    rust \
    linux-headers \
    libc-dev \
    libgcc \
    libstdc++ \
    ca-certificates \
    zlib-dev \
    bzip2-dev \
    xz-dev \
    lz4-dev \
    zstd-dev \
    snappy-dev \
    brotli-dev \
    build-base \
    autoconf \
    boost-dev \
    flex \
    libxml2-dev \
    libxslt-dev \
    libjpeg-turbo-dev \
    ninja \
    git \
    && pip install --upgrade pip && pip install pipenv cython numpy

ARG ARROW_VERSION=17.0.0
ARG ARROW_SHA256=8379554d89f19f2c8db63620721cabade62541f47a4e706dfb0a401f05a713ef
ARG ARROW_BUILD_TYPE=release

ENV ARROW_HOME=/usr/local \
    PARQUET_HOME=/usr/local

RUN mkdir /arrow \
    && wget -q https://github.com/apache/arrow/archive/apache-arrow-${ARROW_VERSION}.tar.gz -O /tmp/apache-arrow.tar.gz \
    && echo "${ARROW_SHA256} *apache-arrow.tar.gz" | sha256sum /tmp/apache-arrow.tar.gz \
    && tar -xvf /tmp/apache-arrow.tar.gz -C /arrow --strip-components 1

# Create the patch file for re2
RUN echo "diff --git a/util/pcre.h b/util/pcre.h" > /arrow/re2_patch.diff \
    && echo "index e69de29..b6f3e31 100644" >> /arrow/re2_patch.diff \
    && echo "--- a/util/pcre.h" >> /arrow/re2_patch.diff \
    && echo "+++ b/util/pcre.h" >> /arrow/re2_patch.diff \
    && echo "@@ -21,6 +21,7 @@" >> /arrow/re2_patch.diff \
    && echo " #include \"re2/filtered_re2.h\"" >> /arrow/re2_patch.diff \
    && echo " #include \"re2/pod_array.h\"" >> /arrow/re2_patch.diff \
    && echo " #include \"re2/stringpiece.h\"" >> /arrow/re2_patch.diff \
    && echo "+#include <cstdint>" >> /arrow/re2_patch.diff

# Configure the build using CMake
RUN cd /arrow/cpp \
    && cmake --preset ninja-release-python

# Pre-fetch dependencies without building
RUN cd /arrow/cpp \
    && cmake --build . --target re2_ep -- -j1 || true

# Apply the patch to re2 after the dependencies are fetched but before the build
RUN cd /arrow/cpp/re2_ep-prefix/src/re2_ep \
    && patch -p1 < /arrow/re2_patch.diff

# Continue with the build and install Apache Arrow
RUN cd /arrow/cpp \
    && cmake --build . --target install \
    && rm -rf /arrow /tmp/apache-arrow.tar.gz

# Set the working directory
WORKDIR /app

# Copy only the requirements file to leverage Docker cache
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY app.py /app/

# Expose the required port
EXPOSE 8501

# Set env
ENV PYTHONUNBUFFERED=1

# Define the entrypoint to run the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
