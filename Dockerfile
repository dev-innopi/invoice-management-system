# Use official Python runtime as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .
# Expose port for FastAPI
EXPOSE 8000
# Create start script
COPY <<EOF /start.sh
#!/bin/bash
# Start the FastAPI application using uvicorn CLI
fastapi run server.py --host 0.0.0.0 --port 8000
EOF

RUN chmod +x /start.sh
CMD ["/start.sh"]