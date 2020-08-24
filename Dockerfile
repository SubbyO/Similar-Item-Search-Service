FROM python:3.7.1
RUN apt-get update
RUN apt-get install -y libsnappy-dev
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 13000
ENTRYPOINT ["python","./app/service.py","--datadir","./app/data"]