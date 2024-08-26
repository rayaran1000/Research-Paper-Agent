# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the dependencies specified in the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the files from the current directory to the working directory
COPY . .

# Specify the command to run the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8000", "--server.address=0.0.0.0"]
