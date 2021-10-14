# Creating docker image for the API

In this document we are going to explain how to create the docker image from the API.

The main file is [Dockerfile](https://github.com/edusalcas/single-cell-repo/blob/main/docker-image/Dockerfile). With this file we will be able to create the image. Here we can see the content of the file:

```dockerfile
FROM python:3

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
```

We can see the image is created from the Python3 docker image. In the [requirements](https://github.com/edusalcas/single-cell-repo/blob/main/docker-image/requirements.txt) file we have all the python libraries we use in the API. Finally, we need a tar document in which we have the [API folder](https://github.com/edusalcas/single-cell-repo/tree/main/API-REST).

Once we have the Dockerfile, the requeriment file and the compressed file in the same folder we can create the docker image with the command:

`docker build --tag image_name:image_tag .`

Indicating the name and the tag of the image.
