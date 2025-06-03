# configs/celeryconfig.py
import os

task_serializer = 'json'
accept_content = ['json']
result_serializer = 'json'
timezone = 'UTC'
enable_utc = True
task_track_started = True
task_time_limit = 3600
task_soft_time_limit = 3600
worker_concurrency = 4
broker_url = os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0')
result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')