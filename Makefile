deps:
	pip install -r requirements.txt

kaya:
	python3 main_kaya.py

loop:
	python3 kaya_loop_audio_files.py

spectrogram:
	python3 kaya_make_spectrogram.py

samples_count:
	ls -1 audio  | wc -l

samples_size:
	du -h audio
