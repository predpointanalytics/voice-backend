from fastapi import FastAPI, status, HTTPException, Request
import psycopg2
from psycopg2.extras import RealDictCursor
import time
from pydantic import BaseModel
import random
import pickle
import pandas as pd
from speechbrain.pretrained import EncoderClassifier
from scipy.spatial.distance import cdist
from speechbrain.dataio.preprocess import AudioNormalizer
import torchaudio
a_norm= AudioNormalizer()
import subprocess

app = FastAPI()

classifier = EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir='model')

try:
     with open('appv2_pkl', 'rb') as files:
        x= pickle.load(files)
        print('df loaded')
except Exception as err:
    print(err)

class Post(BaseModel):
    title: str
    content: str
    published: bool = True


@app.patch('/upload')
async def get_post(request: Request):
    post_data = await request.body()
    audio_file = open("temp22.m4a", "wb")
    audio_file.write(post_data)
    subprocess.call(['ffmpeg', '-i', 'temp22.m4a', 'temp22.wav', '-y'])
    # print(post_data)
    signal, fs =torchaudio.load('temp22.wav', channels_first=False)
    signal = a_norm(signal, fs)
    emb =  classifier.encode_batch(signal)
    comp= x[x.rec_id==1]['embedding'].item()
    score = cdist(emb[0], comp, metric='cosine')
    # print(emb[0])
    ref_id = len(x)+1
    x.loc[ref_id] = [ref_id, emb[0]]
    try:
     with open('appv2_pkl', 'wb') as files:
        pickle.dump(x,files)
        print('df saved')
    except Exception as err:
        print(err) 
    # return random.randint(1,9)
    return ref_id

@app.patch('/verify/{id}')
async def get_post( id: int, request: Request):
    # id=2
    print(id)
    post_data = await request.body()
    audio_file = open("verify.m4a", "wb")
    audio_file.write(post_data)
    subprocess.call(['ffmpeg', '-i', 'verify.m4a', 'verify.wav', '-y'])

    signal, fs =torchaudio.load('verify.wav', channels_first=False)
    signal = a_norm(signal, fs)
    emb =  classifier.encode_batch(signal)
    comp= x[x.rec_id==id]['embedding'].item()
    score = cdist(emb[0], comp, metric='cosine')
    print(score)
    # print(x)
    if score < 0.6:
        return f'Authenticated'
    else:
        return 'Wrong User'

@app.get('/')
def root():
    # cursor.execute("""SELECT * from posts""")
    # posts = cursor.fetchall()
    return {'data': 'Works fine yo'}