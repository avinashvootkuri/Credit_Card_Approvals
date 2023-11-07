FROM python:3.9.18-slim
# FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile","Pipfile.lock","./"]

RUN pipenv install --deploy --system

COPY ["predict.py", "model_eta=0.01.bin", "./"]

EXPOSE 9698

ENTRYPOINT [ "gunicorn","--bind=0.0.0.0:9695","predict:app"]
