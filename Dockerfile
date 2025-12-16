# Use a lightweight Python image as the base
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
# Assuming your Streamlit app code is in a file like 'app.py'
# You would replace 'app.py' with the actual name of your Streamlit app file
COPY . .

# Expose the port that Streamlit runs on (default is 8501)
EXPOSE 8501

# Command to run the Streamlit application
# Make sure 'app.py' matches the name of your main Streamlit file
ENTRYPOINT ["streamlit", "run", "clean.py", "--server.port=8501", "--server.address=0.0.0.0"]
