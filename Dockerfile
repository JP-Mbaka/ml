# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# clear pip purge
# RUN pip cache purge

# Install any needed dependencies specified in requirements.txt
RUN pip install -r requirements.txt

# Install cron
RUN apt-get update && apt-get install -y cron nano

# Create a log file for cron jobs
RUN touch /var/log/cron.log

# Add a cron job to execute teddy.py
RUN echo "0 0 * * * /usr/local/bin/python /app/ml_script.py >> /var/log/cron.log 2>&1" > /etc/cron.d/ml_cron

# Apply cron job permissions and load it
RUN chmod 0644 /etc/cron.d/ml_cron && crontab /etc/cron.d/ml_cron

# Expose port 8008 to the outside world
EXPOSE 8008

# Start cron and app.py
CMD ["sh", "-c", "cron && python app.py"]

