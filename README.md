## Cleo

Build and Test predictive HMM Models that forecast future stock ticker values

#### Steps to run

1. Install requirements `pip install -r requirements.txt`
2. Start up a Postgres DB with `postgres -D /usr/local/var/postgres`
3. Start Flask server `flask run`
4. Start a redis server with `redis-server`
5. Start the rq worker with `rq worker cleo-tasks`

#### Useful commands

`heroku redis:cli -a cleo-stock` Access the current remote redis:cli `FLUSHALL`
