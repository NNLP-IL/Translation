FROM python:3.11 

WORKDIR /app

COPY ../ /app

RUN cat requirements.txt && pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["python3.11", "main.py"]