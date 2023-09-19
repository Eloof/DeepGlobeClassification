FROM python:3.10

RUN apt-get update

WORKDIR /opt/app

ENV PYTHONPATH '/opt/app'

COPY pyproject.toml ./
COPY poetry.lock ./
RUN apt-get update \
    && pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-dev

COPY src src
COPY dataLoader dataLoader
COPY configs configs
COPY data/Data/class_dict.csv data/Data/class_dict.csv


CMD ["python", "src/CustomUnetNN/predict_py.py"]