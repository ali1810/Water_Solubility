<<<<<<< HEAD
FROM python:3.9.5-slim

# Set environment variables to prevent .pyc files and buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy app files
COPY . /app
WORKDIR /app

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "main1.py", "--server.port=8501", "--server.address=0.0.0.0"]

=======
# Use an official lightweight Python image as the base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the contents of the current directory (your project folder) into the container
COPY . /app

# Upgrade pip to the latest version to avoid version-related issues
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies with retries and increased timeout
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Expose the default Streamlit port
EXPOSE 8501

# Set the Streamlit command as the default entry point
ENTRYPOINT ["streamlit", "run"]

# Specify your main Streamlit app file
CMD ["main1.py"]
>>>>>>> e325065 ( Fisrt commit)
