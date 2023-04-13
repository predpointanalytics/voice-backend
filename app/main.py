from fastapi import FastAPI, status, HTTPException, Request
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
import time
from pydantic import BaseModel
import pandas as pd
from speechbrain.pretrained import EncoderClassifier
from scipy.spatial.distance import cdist
from speechbrain.dataio.preprocess import AudioNormalizer
import torchaudio
a_norm= AudioNormalizer()
import subprocess
import io
import torch
import numpy

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
                                user='postgres', 
                                password='digi@post123',
                                # password='password',
                                cursor_factory=RealDictCursor)
    print('DB connection successful!')
    # cursor = conn.cursor()
    return conn



@app.patch('/upload')
async def get_post(request: Request):
    conn = init()
    cursor = conn.cursor()
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

    if request.headers['val']:
        print(request.headers['val'])
        name = request.headers['val']
    else: print('Name not set')

    try:
        cursor.execute("""INSERT INTO embeddings_v2 (embeddings,name) VALUES (%s,%s)
        RETURNING *""", (psycopg2.Binary(data),name))
        new_post =  cursor.fetchone()
        buffer = io.BytesIO(new_post['embeddings'])
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
    conn = init()
    cursor = conn.cursor()
    print(id)
    post_data = await request.body()
    audio_file = open("verify.m4a", "wb")
    audio_file.write(post_data)
    subprocess.call(['ffmpeg', '-i', 'verify.m4a', 'verify.wav', '-y'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
    
##############################################################################
@app.patch('/verifymany')
async def verify(request: Request):
    print('Checking DB connection')
    conn = init()
    cursor = conn.cursor()
    # id =10

    post_data = await request.body()
    print(post_data)
    audio_file = open("verify.m4a", "wb")
    audio_file.write(post_data)
    subprocess.call(['ffmpeg', '-i', 'verify.m4a', 'verify.wav', '-y'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    signal, fs =torchaudio.load('verify.wav', channels_first=False)
    signal = a_norm(signal, fs)
    emb =  classifier.encode_batch(signal)
    print(emb[0].shape)
    # conn.close()
    try:
        cursor.execute("""SELECT name,embeddings from embeddings_v2""")
        bin_list= cursor.fetchall()

        x_df = pd.DataFrame(bin_list)
        x_df['score']= x_df['embeddings'].apply(lambda x: cdist(torch.load(io.BytesIO(x)), emb[0], metric='cosine'))
        # name_df= x_df.iloc[x_df['score'].idxmin()]
        # print(name_df)
        
        min_score= x_df.score.min().item()
        name_df = x_df[x_df.score == min_score]['name'].item()
        print(name_df)
        print(min_score)
        print(x_df)
        
        cursor.close()
        conn.close()
    except Exception as err:
        print('error: ', err)
        cursor.close()
        conn.close()
    ret_mesg = 'Authorized'
    if name_df:
        ret_mesg = 'Welcome ' + name_df
    if min_score < 0.6:
        return ret_mesg
    else: return 'Access Denied'

##############################################################################
#Trying out the new pgvector extension to calculate cosine similarity in db ##
##############################################################################
##############################################################################

@app.patch('/uploadv2')
async def get_post(request: Request):
    conn = init()
    cursor = conn.cursor()
    register_vector(cursor)
    post_data = await request.body()
    audio_file = open("temp22.m4a", "wb")
    audio_file.write(post_data)
    subprocess.call(['ffmpeg', '-i', 'temp22.m4a', 'temp22.wav', '-y'])
    # print(post_data)
    signal, fs =torchaudio.load('temp22.wav', channels_first=False)
    signal = a_norm(signal, fs)
    emb =  classifier.encode_batch(signal)
    data = numpy.array(emb[0][0])

    # buffer = io.BytesIO()
    # torch.save(emb[0], buffer)
    # data = buffer.getvalue()

    if request.headers['val']:
        print(request.headers['val'])
        name = request.headers['val']
    else: print('Name not set')

    try:
        cursor.execute("""INSERT INTO embeddings_v3 (embeddings,name) VALUES (%s,%s)
        RETURNING *""", (data,name))
        new_post =  cursor.fetchone()
        # buffer = io.BytesIO(new_post['embeddings'])
        # arr = torch.load(buffer) 
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

@app.patch('/verifymanyv2')
async def verify(request: Request):
    print('Checking DB connection')
    conn = init()
    cursor = conn.cursor()
    # id =10
    register_vector(cursor)
    post_data = await request.body()
    # print(post_data)
    audio_file = open("verify.m4a", "wb")
    audio_file.write(post_data)
    subprocess.call(['ffmpeg', '-i', 'verify.m4a', 'verify.wav', '-y'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    signal, fs =torchaudio.load('verify.wav', channels_first=False)
    signal = a_norm(signal, fs)
    emb =  classifier.encode_batch(signal)
    data = numpy.array(emb[0][0])
    # print(emb[0].shape)
    # conn.close()
    try:
        cursor.execute(""" SELECT name, embeddings <=> %s as score FROM embeddings_v3 """, (data, ))
        # cursor.execute(""" SELECT * FROM embeddings_v3""")
        emb_result= cursor.fetchone()

        print('zzzzzzzzzzzz',emb_result['name'])

        
        cursor.close()
        conn.close()
        return (emb_result['name'], emb_result['score'])
    except Exception as err:
        print('error: ', err)
        cursor.close()
        conn.close()
    # ret_mesg = 'Authorized'
    # if name_df:
    #     ret_mesg = 'Welcome ' + name_df
    # if min_score < 0.6:
    #     return ret_mesg
    # else: return 'Access Denied'
    # return emb_result['name']
    return type(data)



##############################################################################
##############################################################################
@app.get('/')
def root():
    # cursor.execute("""SELECT * from posts""")
    # posts = cursor.fetchall()
    return {'data': 'Works fine yo!!!!'}