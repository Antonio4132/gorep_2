FROM python:3.7

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY files.tar ./
RUN tar -xf files.tar && \
	rm files.tar

ENV FLASK_APP "entrypoint:app"
ENV FLASK_ENV "development"
ENV APP_SETTINGS_MODULE "config.default"

WORKDIR /usr/src/app/API-REST

CMD flask run -h '0.0.0.0'