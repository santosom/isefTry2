**Commands to commit source code:
```
git status .
git add -A .
git commit -m "your comment"
git push
```

# Setup git-lsf

Download from https://git-lfs.com
Add to your path
Run git-lfs install
git lfs track audio

# Process to classify

* Save the audio file
* Make a spectrogram
* Post-process the spectrogram to grayscale, reduced colors and cropped
* Classify with the the saved model**
