# Base image
FROM python:3.11-slim

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlops_dtu/ mlops_dtu/
COPY data/ data/
# COPY model/ model/
# COPY reports/ reports/

WORKDIR /

RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENV PYTHONPATH /

ENTRYPOINT ["python", "-u", "mlops_dtu/train_model.py"]