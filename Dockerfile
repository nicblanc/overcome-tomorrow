FROM python:3.10.6

COPY overcome_tomorrow /overcome_tomorrow
COPY requirements.txt requirements.txt
COPY Makefile Makefile
COPY setup.py setup.py

RUN pip install --upgrade pip
RUN pip install -e .
RUN mkdir -p /model
RUN mkdir -p /data

CMD uvicorn --host 0.0.0.0 --port $PORT overcome_tomorrow.api.overcome_api:tomorrow_app
