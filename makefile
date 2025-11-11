sync:
	uv sync

run:
	cd app && uvicorn main:app --reload

docker-build:
	docker build -t fastapi-tutorial .

docker-run:
	docker run -p 8000:8000 fastapi-tutorial