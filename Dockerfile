FROM python:3.10.9

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
