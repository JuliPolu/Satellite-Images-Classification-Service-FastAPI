from dependency_injector import containers, providers

from src.services.satellite_classifier import SatelliteClassifier


class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    satellite_classifier = providers.Factory(
        SatelliteClassifier,
        config=config.services.satellite_classifier,
    )
