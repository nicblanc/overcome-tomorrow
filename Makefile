.DEFAULT_GOAL := run_api
reinstall_package:
	@pip uninstall -y overcome-tomorrow || :
	@pip install -e .


run_api:
	uvicorn --host 0.0.0.0 --port $$PORT overcome_tomorrow.api.overcome_api:tomorrow_app --reload
