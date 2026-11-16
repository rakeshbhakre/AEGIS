"""FastAPI application construction."""
from fastapi import FastAPI


def create_app() -> FastAPI:
    return FastAPI(title="AEGIS mission console", version="0.1")


app = create_app()
