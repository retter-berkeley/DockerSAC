# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/coax-dev/coax.git@main --quiet

# Run app.py when the container launches
ENV SAGEMAKER_PROGRAM sacapp.py