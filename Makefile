.PHONY: start stop logs rebuild test-gpu verify-container-assets

start:
	docker compose up -d

stop:
	docker compose down

logs:
	docker compose logs -f worker

rebuild:
	docker compose build --no-cache worker

test-gpu:
	docker run --rm --gpus device=1 nvidia/cuda:12.2.0-runtime-ubuntu22.04 nvidia-smi

verify-container-assets:
	bash scripts/verify_container_assets.sh
