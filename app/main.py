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
import io
import torch

app = FastAPI()

classifier = EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir='model')


# while True:
#     try:
#         print('pre db load 0.2')
#         conn = psycopg2.connect(host='localhost', database='fastapi',
#                                 user='postgres', password='password',
#                                 cursor_factory=RealDictCursor)
#         print('DB connection successful!')
#         cursor = conn.cursor()
#         break
#     except Exception as error:
#         print('Connection Failed')
#         print('Error: ', error)
#         time.sleep(2)

def init():
    conn = psycopg2.connect(host='localhost', database='fastapi',
                                user='postgres', password='password',
                                cursor_factory=RealDictCursor)
    print('DB connection successful!')
    cursor = conn.cursor()
    return (conn, cursor)



@app.patch('/upload')
async def get_post(request: Request):
    conn, cursor = init()
    post_data = await request.body()
    audio_file = open("temp22.m4a", "wb")
    audio_file.write(post_data)
    subprocess.call(['ffmpeg', '-i', 'temp22.m4a', 'temp22.wav', '-y'])
    # print(post_data)
    signal, fs =torchaudio.load('temp22.wav', channels_first=False)
    signal = a_norm(signal, fs)
    emb =  classifier.encode_batch(signal)

    buffer = io.BytesIO()
    torch.save(emb[0], buffer)
    data = buffer.getvalue()

    try:
        cursor.execute("""INSERT INTO embeddings (embs) VALUES (%s)
        RETURNING *""", (psycopg2.Binary(data),))
        new_post =  cursor.fetchone()
        buffer = io.BytesIO(new_post['embs'])
        arr = torch.load(buffer) 
        # print(arr)
        # print(new_post['id'])
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as err:
        print('error: ', err)
        cursor.close()
        conn.close()
    
    if not new_post:
            return 'Upload failed'
    

    return new_post['id']

@app.patch('/verify/{id}')
async def get_post( id: int, request: Request):
    # id=2
    conn, cursor = init()
    print(id)
    post_data = await request.body()
    audio_file = open("verify.m4a", "wb")
    audio_file.write(post_data)
    subprocess.call(['ffmpeg', '-i', 'verify.m4a', 'verify.wav', '-y'])

    signal, fs =torchaudio.load('verify.wav', channels_first=False)
    signal = a_norm(signal, fs)
    emb =  classifier.encode_batch(signal)
    try:
        cursor.execute("""SELECT embs from embeddings WHERE id={0}""".format(id))
        bin_data = cursor.fetchone()['embs']
        buffer = io.BytesIO(bin_data)
        emb_db = torch.load(buffer)

        cursor.close()
        conn.close()
        # print(emb_db)
    except Exception as err:
        print('error: ', err)
        cursor.close()
        conn.close()
    
    if not bin_data:
        return 'Enter valid id'
    score = cdist(emb[0], emb_db, metric='cosine')
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
    return {'data': 'Works fine yo!!!!'}