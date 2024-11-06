FROM python:3.10.12-slim

WORKDIR /home

COPY ./docker/api/requirements_be.txt ./
COPY ./src ./src/
COPY ./models ./models/
COPY ./config ./config/

RUN \
apt-get update && \
apt-get upgrade -y && \
apt-get autoremove -y && \
apt-get clean -y && \
pip install --upgrade pip && \
pip install wheel && \
pip install -r requirements_be.txt

EXPOSE 8080

CMD ["python", "src/api.py"]