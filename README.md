# FastAPI Application for Satelite Images classification (Pet Project)

FastAPI Сервис по мультилейбл классификации космических снимков.
Данный репозиторий подготовлен в рамках 1-й проектной работы курса 'CV-Rocket' от команды Deepschool, часть по созданию сервиса в парадигме GitLab CI/CD.


## Используемый стэк

FastAPI, Pytorch, albumentations, pydantic, pytest, wemake(flake8), OpenCV, omegaconf, dvc, ansible, Docker


### Ссылка на сервис (временная)

[Временная Ссылка](http://91.206.15.25:1667/docs)


### Описание API, примеры запросов

Cервис поднят с помощью приложения FastAPI

Реализованы следующие запросы:

`'GET' '/satellite/classes'` - получение общего списка классов

 `'POST' '/satellite/predict_proba'` - Предсказание классов космических снимков, принимает на вход изображение

`'POST' '/satellite/predict'`  - редсказание вероятностей принадлежности к классам cнимков, принимает на вход изображение

`'GET' '/satellite/health_check'`  - Проверка работает ли сервис


### Как развернуть сервис локально питоном 

* Сначала создать и активировать venv:

```bash
python3 -m venv venv
. venv/bin/activate
```
* Поставить зависимости:
```bash
make install
```

* Cкачать веса моделек
```bash
make download_weights
```

* Запуск сервиса
```bash
make run_app
```
Можно с аргументом `APP_PORT`


### Как развернуть сервис локально докером

* `make build` - собрать образ. Можно с аргументами `DOCKER_TAG`, `DOCKER_IMAGE`

* `make docker_run` - запустить контейнер с приложением


### Где искать докер-образы сервиса и как сбилдить свежий образ

`registry.deepschool.ru/cvr-aug23/j.polushina/hw-01-service`


### Статический анализ - Линтеры

* `make lint` - запуск линтеров


#### Тестирование
* `make run_unit_tests` - запуск юнит-тестов
* `make run_integration_tests` - запуск интеграционных тестов
* `make run_all_tests` - запуск всех тестов
* `make generate_coverage_report` - сгенерировать coverage-отчет
