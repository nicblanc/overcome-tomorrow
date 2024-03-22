.DEFAULT_GOAL := run_api
reinstall_package:
	@pip uninstall -y overcome-tomorrow || :
	@pip install -e .

build_dev_image:
	docker build --tag=$$IMAGE_NAME:dev .

run_dev_image:
	docker run -e PORT=$$PORT -p $$PORT:$$PORT --env-file .env $$IMAGE_NAME:dev

build_and_run_dev_image: build_dev_image run_dev_image

build_prod_image:
	docker build --tag=$$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$GCP_REPOSITORY/$$IMAGE_NAME:prod .

create_gcp_repository:
	- gcloud artifacts repositories create $$GCP_REPOSITORY --repository-format=docker --location=$$GCP_REGION --description="Repository for storing overcome-tomorrow images"

push_image:
	docker push $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$GCP_REPOSITORY/$$IMAGE_NAME:prod

deploy:
	gcloud run deploy --image $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$GCP_REPOSITORY/$$IMAGE_NAME:prod --memory $$GAR_MEMORY --region $$GCP_REGION --env-vars-file .env.yaml

build_and_deploy_image: build_prod_image create_gcp_repository push_image deploy

streamlit:
	streamlit run overcome_tomorrow/api/frontend_file.py

run_api:
	uvicorn --host 0.0.0.0 --port $$PORT overcome_tomorrow.api.overcome_api:tomorrow_app --reload

upload_files_to_bq:
	python -c 'from overcome_tomorrow.utils.data import upload_csv_to_bq; upload_csv_to_bq("raw_data/activities.csv"); upload_csv_to_bq("raw_data/garmin_data.csv")'
