#FROM python:3.9.12

FROM tensorflow/tensorflow:2.6.1

RUN mkdir -p /app

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt

#RUN python -m nltk.downloader stopwords

# RUN python -m nltk.downloader punkt && \
#     python -m nltk.downloader averaged_perceptron_tagger && \
#     python -m nltk.downloader wordnet && \
#     python -m nltk.downloader omw-1.4

EXPOSE 5000             
#only flask


COPY ./nltk_data /root/nltk_data

ENV FLASK_APP=app.py

COPY . /app

#CMD ["python", "app.py"]

ENTRYPOINT [ "flask"]

CMD ["run", "--host", "0.0.0.0"]
#CMD ["gunicorn", "app:app", "--timeout 40"] #for heroku