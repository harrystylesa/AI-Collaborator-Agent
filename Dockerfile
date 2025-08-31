# Use an official Python runtime as a parent image
FROM python:3.13-slim


FROM python:3.10
COPY . .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 80
ENTRYPOINT [ "python" ]
CMD [ "main.py" ]