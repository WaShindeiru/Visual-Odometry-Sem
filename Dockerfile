FROM ros:noetic-ros-core
LABEL authors="washindeiru"

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    git \
    python3-pip

COPY requirements.txt requirements.txt

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install -r requirements.txt

ENTRYPOINT ["/bin/bash"]