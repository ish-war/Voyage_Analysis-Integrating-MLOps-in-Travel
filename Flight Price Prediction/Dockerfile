# Use Python 3.11 slim image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the required files
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app files
COPY . .

# Expose the port (optional, if using Flask)
EXPOSE 5000

# Define the command to run the app
CMD ["python", "app.py"]
