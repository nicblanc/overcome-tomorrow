from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

tomorrow_app = FastAPI()
tomorrow_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@tomorrow_app.get("/")
def root():
    return {"Overcome": "Tomorrow"}
