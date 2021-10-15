web: flask db upgrade; gunicorn cleo:app --timeout 0
worker: rq worker -u $REDIS_URL cleo-tasks