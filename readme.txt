To run a local copy of this app take the following steps:

1. Set up a Virtual Machine with Ubuntu 16.04
2. Install Python 2.7
3. Install RabbitMQ (sudo apt-get install rabbit-server)
4. Start RabbitMQ (service rabbitmq-server start)
5. Install pip (sudo apt-get install python-pip)
6. Pull down code from this repo (git clone https://github.com/michaelnachbar/turfclusteringapp)
7. Install the requirements (pip install -r requirements.txt)
8. Start the local server (python manage.py runserver)
9. Start a celery instance (celery -A tasks worker --loglevel=info)
10. Go to localhost:8000/cutter and the site should be live.
