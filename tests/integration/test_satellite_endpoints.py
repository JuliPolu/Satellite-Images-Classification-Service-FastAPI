from fastapi.testclient import TestClient
from http import HTTPStatus


def test_classes_list(client: TestClient):
    response = client.get('/satellite/classes')
    assert response.status_code == HTTPStatus.OK

    classes = response.json()['classes']

    assert isinstance(classes, list)


def test_predict(client: TestClient, sample_image_bytes: bytes):
    files = {
        'image': sample_image_bytes,
    }
    response = client.post('/satellite/predict', files=files)

    assert response.status_code == HTTPStatus.OK

    predicted_genres = response.json()['classes']

    assert isinstance(predicted_genres, list)


def test_predict_proba(client: TestClient, sample_image_bytes: bytes):
    files = {
        'image': sample_image_bytes,
    }
    response = client.post('/satellite/predict_proba', files=files)

    assert response.status_code == HTTPStatus.OK

    classes2prob = response.json()

    for classes_prob in classes2prob.values():
        assert classes_prob <= 1
        assert classes_prob >= 0
