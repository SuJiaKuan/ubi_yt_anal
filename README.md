# YouTube UBI Videos Analysis

## Installation

* Install Miniconda or Anaconda.

* Create a Conda environment: `yt_anal`.
```bash
conda create --name yt_anal python=3.11
```

* Activate the environment.
```bash
conda activate yt_anal
```

* Install required packages.
```bash
pip install -r requirements.txt
```

## Retrieve YouTube Comments

* Go to [Google Cloud Console](https://console.cloud.google.com/), create or select a project, and enable the YouTube Data API v3.

* Create `.env` and fill your Google API key in the file.
```
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
```

* Run demo script in command line.
```bash
# Example: https://www.youtube.com/watch?v=_-VNHHtYX3k
python retrieve_yt_comments.py _-VNHHtYX3k
```