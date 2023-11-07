FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

WORKDIR /app

COPY requirements.txt .
COPY requirements_denoiser.txt .

RUN pip3 install scipy
RUN pip3 install "Pillow==8.3.2"
RUN pip3 install setuptools-rust
RUN pip3 install --upgrade pip

RUN pip3 install opencv-python==4.5.3.56 

RUN pip3 install -r requirements_denoiser.txt
RUN pip3 install -r requirements.txt

COPY . .

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
