.PHONY: start stop logs rebuild test-gpu verify-container-assets verify-container-assets-clean check-deps-sync

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

verify-container-assets-clean:
	NO_CACHE=1 RESET=1 bash scripts/verify_container_assets.sh

check-deps-sync:
	python3 scripts/check_deps_sync.py
