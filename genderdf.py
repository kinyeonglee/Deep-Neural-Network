import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from selenium import webdriver
import time
import numpy as np
import urllib3
import cv2
import pickle

def create_df():
    input_file=open('male.dat','rb')
    male_list=pickle.load(input_file)
    input_file=open('female.dat','rb')
    female_list=pickle.load(input_file)

    p=0
    for i,x in enumerate(male_list):
        response=requests.get(x)
        if response.status_code!=200:
            continue
        image = np.asarray(bytearray(response.content))
        imageBGR = cv2.imdecode(image, cv2.IMREAD_COLOR)
        imageRGB = cv2.cvtColor(imageBGR , cv2.COLOR_BGR2RGB)
        imageRGB = cv2.resize(imageRGB, (64, 64))
        imageRGB=imageRGB.reshape(imageRGB.shape[0]*imageRGB.shape[1]*imageRGB.shape[2],1)
        if False not in np.equal(imageRGB[:100],np.zeros(100)):
            continue
        imageRGB = np.append(imageRGB, 1)
        if p==0:
            gender_df=pd.DataFrame(imageRGB)
            p += 1
            continue
        p += 1
        print(f"progress: {format(i/len(male_list)*100/2,'.2f')}%")
        gender_df[f'x{p}']=imageRGB

    for i,x in enumerate(female_list):
        response=requests.get(x)
        if response.status_code!=200:
            continue
        image = np.asarray(bytearray(response.content))
        imageBGR = cv2.imdecode(image, cv2.IMREAD_COLOR)
        imageRGB = cv2.cvtColor(imageBGR , cv2.COLOR_BGR2RGB)
        imageRGB = cv2.resize(imageRGB, (64, 64))
        imageRGB=imageRGB.reshape(imageRGB.shape[0]*imageRGB.shape[1]*imageRGB.shape[2],1)
        if False not in np.equal(imageRGB[:100],np.zeros(100)):
            continue
        imageRGB = np.append(imageRGB, 0)
        p += 1
        print(f"progress: {format((i/len(female_list)*100/2)+50,'.2f')}%")
        gender_df[f'x{p}']=imageRGB
    print('progress: 100%')
    return gender_df
