FROM quay.io/jupyter/scipy-notebook:latest

# Instalar las librerías del proyecto
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /home/jovyan/work
