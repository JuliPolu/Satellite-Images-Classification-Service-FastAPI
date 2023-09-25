import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

from src.containers.containers import AppContainer
from src.routes.routers import router as app_router
from src.routes import satelite as satellite_routes

def create_app() -> FastAPI:
    cfg = OmegaConf.load('config/config.yml')
    container = AppContainer()
    container.config.from_dict(cfg)
    container.wire([satellite_routes])
    app = FastAPI()
    set_routers(app)
    return app


def set_routers(app: FastAPI):
    app.include_router(app_router, prefix='/satellite', tags=['satellite'])

if __name__ == '__main__':
    app = create_app()
    uvicorn.run(app, port=5000, host='0.0.0.0')