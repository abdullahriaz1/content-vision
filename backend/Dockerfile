FROM python:3

WORKDIR /main

COPY . .

RUN pip install 'git+https://github.com/huggingface/transformers.git'

CMD ["python3", "./chat.py"]

EXPOSE 8090