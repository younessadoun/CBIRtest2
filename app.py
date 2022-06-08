import os
#from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import pickle

import string

import cv2 as cv
import numpy as np
import skimage
from matplotlib import pyplot as plt
from skimage import metrics
from skimage import io
#from google.colab.patches import cv2_imshow
import skimage.feature as feature
import pickle
import os
from scipy.spatial import distance

class userr:
    def __init__(self, name, pas):
        self.n = name
        self.p = pas

class index:
    n = None
    c = None
    c2= None
    s = None
    t = None
    p = None
    d = None

def color_moments(filename):
    img = cv.imread(filename)
    if img is None:
        return
    # Convert BGR to HSV colorspace
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv.split(hsv)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    #color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    #color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    #color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])
    color_feature=[h_mean, s_mean, v_mean, h_std, s_std, v_std, h_thirdMoment, s_thirdMoment, v_thirdMoment]
    return color_feature

def extract_features(name):
    print(name)


    img=cv.imread(name)
    #print(img)

    # Calculate histogram without mask


    hist1 = cv.calcHist([img], [0], None, [256], [0, 256])

    #print(hist1)
    colmom = color_moments(name)

    lower = 0.66 * np.mean(img)
    upper = 1.33 * np.mean(img)
    edges = cv.Canny(img, lower, upper)

    # print(edges)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    graycom = feature.graycomatrix(gray, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)

  #  contrast = feature.graycoprops(graycom, 'contrast')
  #  dissimilarity = feature.graycoprops(graycom, 'dissimilarity')
  #  homogeneity = feature.graycoprops(graycom, 'homogeneity')
  #  energy = feature.graycoprops(graycom, 'energy')
  #  correlation = feature.graycoprops(graycom, 'correlation')
  #  ASM = feature.graycoprops(graycom, 'ASM')

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(gray, None)
    #kp = sift.detect(gray, None)



    feat=index
    feat.n = name
    feat.c = hist1
    feat.c2= colmom
    feat.s = edges
    feat.t = graycom
    feat.p = keypoints_1
    feat.d = descriptors_1

    return feat

def search(quer,reg):
    objects = []
    order = []
    #quer='images/test/1.2.826.0.1.3680043.8.498.11678170878548215953866689093584664340-c.png'
    querf=extract_features(quer)
    qc=querf.c
    qc2=querf.c2
    qs=querf.s
    qt=querf.t
    qkp=querf.p
    qd=querf.d
    with (open("data/"+reg+".dat", "rb")) as openfile:
        while True:
            try:
                tem=pickle.load(openfile)

                cd = metrics.hausdorff_distance(qc, tem.c)

                #print(qc2)
                #print(tem.c2)
                cd2=distance.euclidean(qc2,tem.c2)
                #cd2 = metrics.hausdorff_distance(qc, tem.c2)

                #sd = metrics.hausdorff_distance(qs, tem.s)
                sd=cv.matchShapes(qs,tem.s,1,0.0)*10000

                td = metrics.hausdorff_distance(qt, tem.t)

                imgc=cv.imread(tem.n)
                gray = cv.cvtColor(imgc, cv.COLOR_BGR2GRAY)

                sift = cv.SIFT_create()

                keypoints_1, descriptors_1 = sift.detectAndCompute(gray, None)
                #kpc = sift.detect(gray, None)
                bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
                matches = bf.match(descriptors_1, qd)
                kpd=0
                if len(descriptors_1)<=len(qd) :
                    kpd=100-((len(matches)*100)/len(descriptors_1))
                else:
                    kpd =100-( (len(matches) * 100) / len(qd))
             #   print(len(descriptors_1))
             #   print(len(qd))
             #   print(len(matches))
             #   matches = sorted(matches, key=lambda x: x.distance)
                #kpd = metrics.hausdorff_distance(qkp, tem.p)

                ttd = cd*0.5 + cd2*0.5 + sd*1 + td*0.1 + kpd*10
              #  ttd = sd

                o=(ttd,tem.n)

                order.append(o)

                objects.append(tem)
               # print(objects)
            except EOFError:
                break
        print("end")


    order.sort()
    resultlist=[None,None,None,None,None]
    for i in range(5):
        resultlist[i]=order[i][1]

    return resultlist

def emptyF(path):
    import os
    import glob

    files = glob.glob(path)
    for f in files:
        if os.path.isfile(f):
            os.remove(f)


quer=None
region=None

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('login.html')

@app.route('/logged', methods=['POST'])
def logged():
    valid=False
    uname = request.form['uname']
    psw = request.form['psw']
    #print(uname)
    #print(psw)
    with (open("data/users.dat", "rb")) as openfile:
        while True:
            try:
                tem=pickle.load(openfile)
                print(tem.n)
                print(tem.p)
                if uname==tem.n and psw==tem.p:
                    valid=True
                    break
            except EOFError:
                break
    if valid==True:
        return render_template('upload.html')
    else:
        return render_template('login.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        #flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        global region
        region = request.form['region']
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    global quer
    quer = filename
    #print(region)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/my-link')
def my_link():
    print('search started')
    res0 = search(UPLOAD_FOLDER + quer,region)
    print(res0)
    res=[]
    i=0
    for im in res0:
        res.append('static/'+res0[i])
        i=i+1
    #print(res)
    emptyF('static/uploads/*')
    return render_template('result.html', links=res0)

if __name__ == "__main__":
    app.run()