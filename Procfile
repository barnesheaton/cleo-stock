web: flask db upgrade; flask translate compile; gunicorn cleo:app
worker: rq worker -u $REDIS_URL cleo-tasks