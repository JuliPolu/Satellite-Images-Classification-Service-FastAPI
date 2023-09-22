import cv2
import numpy as np
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File

from src.containers.containers import AppContainer
from src.routes.routers import router
from src.services.satellite_classifier import SatelliteClassifier


@router.get('/classes')
@inject
def genres_list(service: SatelliteClassifier = Depends(Provide[AppContainer.satellite_classifier])):  # noqa: B008, WPS404, E501
    return {
        'classes': service.classes,
    }


@router.post('/predict')
@inject
def predict(
    image: bytes = File(),
    service: SatelliteClassifier = Depends(Provide[AppContainer.satellite_classifier]),
):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    classes = service.predict(img)

    return {'classes': classes}


@router.post('/predict_proba')
@inject
def predict_proba(
    image: bytes = File(),
    service: SatelliteClassifier = Depends(Provide[AppContainer.satellite_classifier]),
):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    return service.predict_proba(img)


@router.get('/health_check')
def health_check():
    return 'OK'
