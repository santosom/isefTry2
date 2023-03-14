deps:
	pip install -r requirements.txt

kaya:
	python3 main_kaya.py

loop:
	#rm -rf spectrograms
	#mkdir spectrograms
	python3 kaya_loop_audio_files.py

spectrogram:
	python3 kaya_make_spectrogram.py audio/emergency_alarms/sound_1.wav

postprocess:
	python3 post_process.py

importdata:
	python3 import_dataset.py

samples_count:
	ls -1 audio  | wc -l

train:
	python3 kaya_train.py

samples_size:
	du -h audio

nn:
	python3 cnn.py
	open results.png

main:
	python3 main.py

keras_model:
	python3 keras_model.py

pydot:
	pip install pydot

smalldataset:
	python3 small_dataset.py

restoredataset:
	git checkout final_spectrograms 

live:
	python3 live.py

play:
	afplay test.wav

