web: flask db upgrade; gunicorn cleo:app
worker: rq worker -u $REDIS_URL cleo-tasks