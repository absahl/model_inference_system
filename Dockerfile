FROM ubuntu:latest

RUN apt update -y
RUN apt install -y python3

WORKDIR /app

CMD ["bash"]
