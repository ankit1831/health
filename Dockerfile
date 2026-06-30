# Use the official Python 3.9 image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all your backend files into the container
COPY . .

# Expose port 7860 (This is the port Hugging Face Spaces requires)
EXPOSE 7860

# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]