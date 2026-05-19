# Use the official Python 3.9 slim image as the base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Prevent Python from writing pyc files to disc and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# 1. Upgrade pip first for better network and dependency handling
RUN pip install --no-cache-dir --upgrade pip

# 2. Add extreme timeouts (1000s) and 10 retries to survive network drops
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]