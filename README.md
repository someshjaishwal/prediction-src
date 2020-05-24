## ReadMe

Steps 1: Clone This Repo :

        git clone https://github.com/someshjaishwal/prediction-src.git
    
Step 2: Create virtual environment:

        cd prediction-src
        virtualenv -p python3 venv
        source venv/bin/activate
    
Step 3: Install dependencies:

        cat requirements.txt | xargs -n 1 pip install
    

## Predict transcript of a single wav file
To predict transcription of an audio named `audio.wav`, use below command from `prediction-src/` directory:

    CUDA_VISIBLE_DEVICE=0 python predict.py --cuda --model-path models/network.pth --audio-path audio.wav
    

## Evaluate the pretrained model using `manifest.csv`
You can also use a manifest.csv file to evaluate the pretrained model. Each row of the manifest.csv file contains path of an audio and transcript of that audio, separated by comma(,) delimiter. 
e.g. : 

    ../data/speaker1/audio1.wav,chilli flakes
    ../data/speaker1/audio2.wav,mint mayonnaise
    
NOTE: Every character of the transcription is small. If you're using `manifest.csv`, make sure spelling of each transcription in `manifest.csv` matches with `files/transcriptions.py`

Finally, to evaluate the pretrained model using `manifest.csv`, hit the below command from `prediction-src/` directory

    CUDA_VISIBLE_DEVICES=0 python evaluate.py --cuda --model-path models/network.pth --manifest manifest.csv 
    


<hr>

#### Needless to mention, at any time we choose CUDA_VISIBLE_DEVICES of our choice