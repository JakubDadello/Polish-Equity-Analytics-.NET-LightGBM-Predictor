# Use Python 3.11 slim image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
# English: Using --no-cache-dir to minimize image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Run the stacking training script as the entry point
CMD ["python", "train_stacking.py"]