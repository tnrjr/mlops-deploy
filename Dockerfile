FROM python:3.7-slim

COPY ./requirements.txt /usr/requirements.txt

WORKDIR /usr

RUN pip3 install -r requirements.txt

COPY ./src /usr/src
COPY .models /usr/model

ENTRYPOINT [ "python3" ]

CMD [ "src/app/main.py" ]