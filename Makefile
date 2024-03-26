.DEFAULT_GOAL := run_api
reinstall_package:
	@pip uninstall -y overcome-tomorrow || :
	@pip install -e .

build_dev_image:
	docker build --build-arg data_path=$$DATA_PATH --build-arg model_path=$$MODEL_PATH --build-arg model_name=$$MODEL_NAME --build-arg activity_preproc_name=$$ACTIVITY_PREPROC_NAME --build-arg garmin_data_preproc_name=$$GARMIN_DATA_PREPROC_NAME -f Dockerfile_dev --tag=$$IMAGE_NAME:dev .

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
	gcloud run deploy  overcome-tomorrow --image $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$GCP_REPOSITORY/$$IMAGE_NAME:prod --memory $$GAR_MEMORY --region $$GCP_REGION --env-vars-file .env.yaml

build_and_deploy_image: build_prod_image create_gcp_repository push_image deploy

streamlit:
	streamlit run overcome_tomorrow/api/frontend_file.py

run_api:
	uvicorn --host 0.0.0.0 --port $$PORT overcome_tomorrow.api.overcome_api:tomorrow_app

run_api_with_reload:
	uvicorn --host 0.0.0.0 --port $$PORT overcome_tomorrow.api.overcome_api:tomorrow_app --reload

start_overcome_tomorrow: run_api | streamlit

upload_files_to_bq:
	python -c 'from overcome_tomorrow.utils.data import upload_csv_to_bq; upload_csv_to_bq("raw_data/activities.csv"); upload_csv_to_bq("raw_data/garmin_data.csv")'

upload_model_to_gcs:
	python -c 'from overcome_tomorrow.utils.data import upload_model_to_gcs; upload_model_to_gcs();'

upload_preprocessors_to_gcs:
	python -c 'from overcome_tomorrow.utils.data import upload_preprocessors_to_gcs; upload_preprocessors_to_gcs();'

upload_model_and_preprocessors_to_gcs:
	python -c 'from overcome_tomorrow.utils.data import upload_model_to_gcs, upload_preprocessors_to_gcs; upload_model_to_gcs(); upload_preprocessors_to_gcs()'

downpload_model_and_preprocessors_from_gcs:
	python -c 'from overcome_tomorrow.utils.data import download_model_and_preprocessors_from_gcs; download_model_and_preprocessors_from_gcs();'

download_files_from_bq:
	python -c 'from overcome_tomorrow.utils.data import save_csv_from_bq; save_csv_from_bq("test_activities.csv", "activities"); save_csv_from_bq("test_garmin_data.csv", "garmin_data")'
