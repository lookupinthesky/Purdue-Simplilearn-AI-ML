FROM python:3.9.12

RUN mkdir -p /app

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt

RUN python -m nltk.downloader punkt && \
    python -m nltk.downloader stopwords && \
    python -m nltk.downloader averaged_perceptron_tagger && \
    python -m nltk.downloader wordnet && \
    python -m nltk.downloader omw-1.4

EXPOSE 5000

ENV FLASK_APP=app.py

COPY . /app

#CMD ["python", "app.py"]

ENTRYPOINT [ "flask"]

CMD ["run", "--host", "0.0.0.0"]