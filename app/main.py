from fastapi import FastAPI, status, HTTPException, Request
import psycopg2
from psycopg2.extras import RealDictCursor
import time
from pydantic import BaseModel
import random
import pickle
import pandas as pd

app = FastAPI()

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
    # print(post_data)
    return random.randint(1,9)
    # print(x)
    return len(x)

@app.get('/')
def root():
    # cursor.execute("""SELECT * from posts""")
    # posts = cursor.fetchall()
    return {'data': 'Works fine yo'}