FROM python:3.10-slim
WORKDIR /app
COPY scripts/ scripts/
RUN pip install pandas scikit-learn joblib

