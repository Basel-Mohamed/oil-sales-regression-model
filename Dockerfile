# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies 
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Cloud Run expects
EXPOSE 8080

# Command to run the application (Adjust if using Flask or a different ASGI/WSGI server)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]