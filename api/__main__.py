"""Run the mission console using configured host and port."""
import uvicorn

from .settings import settings


if __name__ == "__main__":
    uvicorn.run("AEGIS.api.server:app", host=settings.host, port=settings.port)
