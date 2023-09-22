import random
from copy import deepcopy

import cv2
import numpy as np

from src.containers.containers import AppContainer


class FakeSatelliteClassifier:

    def predict(self, image):
        return []

    def predict_proba(self, image):
        return dict(value=0.2)


def test_predicts_not_fail(app_container: AppContainer, sample_image_np: np.ndarray):
    with app_container.reset_singletons():
        with app_container.satellite_classifier.override(FakeSatelliteClassifier()):
            satellite_classifier = app_container.satellite_classifier()
            satellite_classifier.predict_proba(sample_image_np)
            satellite_classifier.predict(sample_image_np)


def test_prob_less_or_equal_to_one(app_container: AppContainer, sample_image_np: np.ndarray):
    with app_container.reset_singletons():
        with app_container.satellite_classifier.override(FakeSatelliteClassifier()):
            satellite_classifier = app_container.satellite_classifier()
            classes2prob = satellite_classifier.predict_proba(sample_image_np)
            for prob in classes2prob.values():
                assert prob <= 1
                assert prob >= 0


def test_predict_dont_mutate_initial_image(app_container: AppContainer, sample_image_np: np.ndarray):
    with app_container.reset_singletons():
        with app_container.satellite_classifier.override(FakeSatelliteClassifier()):
            initial_image = deepcopy(sample_image_np)
            satellite_classifier = app_container.satellite_classifier()
            satellite_classifier.predict(sample_image_np)

            assert np.allclose(initial_image, sample_image_np) 