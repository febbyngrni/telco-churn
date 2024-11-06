FROM python:3.10.12-slim

WORKDIR /home

COPY ./docker/streamlit/requirements_fe.txt ./
COPY ./src ./src/
COPY ./assets ./assets/

RUN \
apt-get update && \
apt-get upgrade -y && \
apt-get autoremove -y && \
apt-get clean -y && \
pip install --upgrade pip && \
pip install wheel && \
pip install -r requirements_fe.txt

EXPOSE 8501

CMD ["streamlit", "run", "src/streamlit.py"]