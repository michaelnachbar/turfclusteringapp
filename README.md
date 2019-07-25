To run a local copy of this app take the following steps:

1. Set up a Virtual Machine with Ubuntu 16.04
1. Install Python 2.7
1. Install RabbitMQ (`sudo apt-get install rabbit-server`)
1. Start RabbitMQ (`service rabbitmq-server start`)
1. Install required packages (`apt install libmysqlclient-dev chromium-chromedriver`)
1. Install pip (`sudo apt-get install python-pip`)
1. Pull down code from this repo (`git clone https://github.com/michaelnachbar/turfclusteringapp`)
1. Install the requirements (`pip install -r requirements.txt`)
1. Setup local_settings.py (contact michael.l.nachbar@gmail.com to get database info)
1. Start the local server (`python manage.py runserver`)
1. Start a celery instance (`celery -A tasks worker --loglevel=info`)
1. Go to localhost:8000/cutter and the site should be live.


To run with sqlite instead of the mysql backend you'll need to create the sqlite database
from a database backup.  The procedure is as follows:
1. Make sure that the sqlite3 and mysql clients installed (`sudo apt-get install sqlite3 mysql-client-5.7`)
1. Make sure the `localsettings.py` in `canvas_cutting` directory is setup to connect to the
   mysql database.
1. From turfclusteringapp repo, run `./tools/mysql2sqlite <(./tools/dump_mysqldb.sh) | sqlite3 canvas_cutter.sqlite`
1. (Optional) Compress large json fields (about 50% reduction in database size) `./tools/sqlite_json_compress.py canvas_cutter.sqlite`
  * If you want to import back to mysql, the compressed fields will have to be decompressed.
1. Edit `local_settings.py` to use the `sqlite3` backend and set `NAME` to `canvas_cutter.sqlite`.
