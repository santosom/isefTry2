deps:
	pip install -r requirements.txt

kaya:
	python3 main_kaya.py

samples_count:
	ls -1 audio  | wc -l

samples_size:
	du -h audio
