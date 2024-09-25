mlflow:
	mlflow server --host 127.0.0.1 --port 8090

run:
	poetry run python src/main.py

debug:
	poetry run python -m pudb src/main.py