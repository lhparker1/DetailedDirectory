#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
from numpy import asarray
from numpy import empty
import os
from os import walk
import PIL
import time
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection
from skimage import exposure
from numpy import linalg
import cv2


"""Edit these variables to adjust model specifications"""
directory = '/Users/liamparker/Desktop/MLAlgo/Images/'
attributes = '/Users/liamparker/Desktop/MLAlgo/attributes.txt'
test_img = '/Users/liamparker/Desktop/MLAlgo/liamtest.jpg'

number_samples = 20
training_split = 0.8
variance_retained = 0.9
load = False
loadPCA = False

feature = 'Male'
grayscale = False
model_type = 'Logistic Regression'
recognize_face = True


"""Add to this dictionary if you want to add more possible features"""
DICTIONARY = dict([
    ('Male', 21),
    ('Attractive', 3),
    ('Black Hair', 9),
    ('Blonde Hair', 10),
    ('Brown Hair', 12),
    ('Chubby', 14),
    ('Pale', 27),
    ('Straight Hair', 33),
    ('Wavy Hair', 34)
    ])

def generate_cutoff(no_face_array, N, training_number):
    train_no_face = 0
    test_no_face = 0
    for val in no_face_array:
        if val < training_number:
            train_no_face += 1
        else:
            test_no_face += 1
    
    cutoff = training_number - train_no_face
    end = N - test_no_face
    
    return cutoff, end
    
@ignore_warnings
def generate_face(test_img):
    img = cv2.imread(test_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier('/Users/liamparker/opt/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    box = face_cascade.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 4)
    
    if box == ():
        return None
    
    x = box[0][0]
    y = box[0][1]
    w = box[0][2]
    h = box[0][3]
    
    HEIGHT = 218
    WIDTH = 178
    
    test_img = Image.open(test_img)
    cropped_img = test_img.crop((x, y, x + w, y + h))
    cropped_img = cropped_img.resize((WIDTH, HEIGHT))
    img = cropped_img
        
    return img

def convert_to_size(test_img, grayscale):
    HEIGHT = 218
    WIDTH = 178
    img = Image.open(test_img)
    img = img.resize((WIDTH, HEIGHT))
    
    if grayscale == True:
        img = img.convert('L')
        img = asarray(img)
        img = np.reshape(img, (WIDTH*HEIGHT))
    if grayscale == False:
        img = asarray(img)
        img = np.reshape(img, (WIDTH*HEIGHT*3))
    else:
        raise ValueError("Grayscale must be boolean")
    return img
 
def load_data(N, split, grayscale):
    
    """
    Loads image data of N samples from a pre-existing centered####.npy file,
    returning train and test  datasets
    """    
    training_number = N*split
    if training_number != int(training_number):
        raise ValueError("N*split must be an integer")
    else:
        training_number = int(training_number)
    
    string = ''
    if grayscale:
        string += 'gray'
    filename ='centered'+str(N)+string+'.npy'
    nofacename='noface'+str(N)+string+'.npy'
    centered = np.load(filename)
    no_face_array = np.load(nofacename)
        
    cutoff, end = generate_cutoff(no_face_array,N,training_number)

    train = centered[:cutoff]
    test = centered[cutoff:end]
    return train, test, no_face_array

def load_PCA(N, grayscale):
    gry = ''
    if grayscale:
        gry += 'gray'
        
    filename = 'PCA'+str(N)+gry+'Train.npy'
    train = np.load(filename)
    filename = 'PCA'+str(N)+gry+'Test.npy'
    test = np.load(filename)
    
    return train, test
 
def import_data(directory, N, split, grayscale, facial_recognition):
    """This processes images from a given directory and returns
    a 3-d array containing all of the image data, after it has been 
    flattened to grayscale"""
    
    training_number = N*split
    if training_number != int(training_number):
        raise ValueError("N*split must be an integer")
    else:
        training_number = int(training_number)
    
    for root,dirs,files in os.walk(directory, topdown = True): 
        files.sort()
        
    files = files[1:]
    
    label = 0
    no_face_array = []  

    sample = asarray(Image.open(directory+files[0])).shape
    HEIGHT = int(sample[0])
    WIDTH = int(sample[1])
    DEPTH = 3
    if grayscale == True:
        img_array = np.empty((N, HEIGHT, WIDTH))
    else:
        img_array = np.empty((N, HEIGHT, WIDTH, DEPTH))
    j = 0
    
    print(len(files))
    for file in files[:N]:
        if facial_recognition == True:
            img = generate_face(directory+file)
            if img == None:
                no_face_array.append(label)
                label = label + 1
                continue
        if facial_recognition == False:
            img = Image.open(directory+file)
            face = generate_face(directory+file)
            if face == None:
                no_face_array.append(label)
                label = label + 1
                continue
        if grayscale == True:
            img = img.convert('L')
        img = asarray(img)
        p2 = np.percentile(img, 2)
        p98 = np.percentile(img, 98)
        data = exposure.rescale_intensity(img, in_range=(p2, p98))
        print(label)
        
        if grayscale == True:
            for i in range(HEIGHT):
                for j in range(WIDTH):
                    img_array[label][i][j] = data[i][j]
            label += 1
        
        else:
            for i in range(HEIGHT):
                for j in range(WIDTH):
                    for k in range(DEPTH):
                        img_array[label][i][j][k] = data[i][j][k]
            label += 1              
    
    i = 0
    for val in no_face_array:
        img_array = np.delete(img_array, val-i, axis=0)
        i=i+1
        
    if grayscale == True:
        img_array = np.reshape(img_array, (N-len(no_face_array), HEIGHT*WIDTH))
    
    else: 
        img_array = np.reshape(img_array, (N-len(no_face_array), HEIGHT*WIDTH*DEPTH))

    
    centered = StandardScaler().fit_transform(img_array)
    gry = ''
    if grayscale:
        gry += 'gray' 
    length = N
    filename = 'centered'+str(N)+gry
    nofacename = 'noface'+str(N)+gry
    np.save(filename, centered)
    np.save(nofacename, no_face_array)
    
    cutoff, end = generate_cutoff(no_face_array,N,training_number)

    train = centered[:cutoff]
    test = centered[cutoff:end]
    return train, test, no_face_array

def Principal_Component_Analysis(variance_retained, train, test, test_img):
    N = train.shape[0] + test.shape[0]
    """Performs PCA with given variance retained"""
    transformer = SparseRandomProjection()
    train_new = transformer.fit_transform(train)
    test_new = transformer.transform(test)

    #test_img_new = transformer.transform(test_img)
    
    train_covariance = np.dot(train_new.T, train_new)
    
    train_val, train_vec = linalg.eigh(train_covariance)
    
    idx = train_val.argsort()[::-1]
    train_val = train_val[idx]
    train_vec = train_vec[:, idx]       
    
    train_variances = []
    for i in range(len(train_val)):
        train_variances.append(train_val[i] / np.sum(train_val))  
    
    sum_var=0
    train_idx = 0
    for i in range(len(train_variances)):
        if sum_var >= variance_retained:
            break
        else:
            sum_var += train_variances[i]
            train_idx += 1
    
    big_train_eigenvectors = train_vec[:,:train_idx]    
    train_data = np.dot(train_new, big_train_eigenvectors)
    test_data = np.dot(test_new, big_train_eigenvectors)
    
    gry = ''
    if grayscale:
        gry += 'gray' 
    filename = 'PCA'+str(N)+gry+'Train'
    np.save(filename, train_data)
    filename = 'PCA'+str(N)+gry+'Test'
    np.save(filename, test_data)
    
    #test_img_new = np.dot(test_img_new, big_train_eigenvectors)
    return train_data, test_data#, test_img_new
    
def process_labels(attributes, N, split, dic, no_face_array):
    """This processes the image labels from the given file 
    and returns a 1d array with all of the appropriate labels
    based on which one was chosen"""
    training_number = N*split
    if training_number != int(training_number):
        raise ValueError("N*split must be an integer")
    else:
        training_number = int(training_number)
    
    attributes_file = attributes 
    file1 = open(attributes_file,"r") 
    st = file1.readlines()

    labels = []
    numLabels = N
    i = 2
    
    col = int(DICTIONARY[dic])
    tf = False
        
    train_no_face = 0
    test_no_face = 0
    
    for val in no_face_array:
        if val < training_number:
            train_no_face += 1
        else:
            test_no_face += 1
    
    while (i < 2 + numLabels):
        if i-2 in no_face_array:
            i = i+1
            continue
        line = st[i].split()
        labels.append(line[col])
        i = i + 1
    
    file1.close()
     
    cutoff = training_number - train_no_face
    end = N - test_no_face
    train_labels = labels[:cutoff]
    test_labels = labels[cutoff:end]
    return train_labels, test_labels

def print_settings():
    print('Settings: ')
    print('feature: '+str(feature))
    print('samples: '+str(number_samples))
    print('load_data: '+str(load))
    print('train-split: '+str(training_split))
    print('grayscale: '+str(grayscale))
    print('recognize-face: '+str(recognize_face))        

@ignore_warnings(category=ConvergenceWarning)
def ML_Model(model_type, train_data, train_label, test_data, test_label):
    """Everything to do with the actual Machine Learning Model""" 
    if model_type == "Logistic Regression":
        clf = LogisticRegressionCV(penalty = 'l2', max_iter = 500, solver = 'saga')
        clf.fit(train_data, train_label)
        score = clf.score(test_data, test_label)
        return clf, score
    else:
        raise ValueError("Invalid Model Type")

def convert_time(tsecs):
    if (tsecs > 3600):
        hours = int(tsecs / 3600)
        timeleft = tsecs - ( hours * 3600)
        minutes = int(timeleft/60)
        seconds = round(timeleft - (minutes * 60))
        string = str(hours) + ' hour(s), ' + str(minutes) + ' minute(s), ' + str(seconds) + ' second(s)'
        
    elif (tsecs > 60):
        minutes = int(tsecs/60)
        seconds = round(tsecs - (minutes * 60))
        string = str(minutes) + ' minute(s), ' + str(seconds) + ' second(s)'
    else:
        string = str(round(tsecs))+' second(s)'
    return string

if __name__ == '__main__':
    "Print model specifications"
    print_settings()
    print()
    
    
    tstart = time.time()
    tlast = tstart

    #test_img = convert_to_size(test_img, True).reshape(1,-1)
    if load:
        train, test, no_face_array = load_data(number_samples, training_split, grayscale)
    else:
        train, test, no_face_array = import_data(directory, number_samples, training_split, grayscale, recognize_face)
    tnow = time.time()
    print('Runtimes: ')
    if load:
        string = 'load_data'
    else:
        string = 'import_data'
    print(string + ': '+str(convert_time(tnow - tlast)))
    tlast = tnow
    
    ', test_img_data'
    if loadPCA:
        length = train.shape[0] + test.shape[0]
        train_data, test_data = load_PCA(length, grayscale)
    else:
        train_data, test_data = Principal_Component_Analysis(variance_retained, train, test, test_img)
    
    
    tnow = time.time()
    if loadPCA:
        string = 'load_PCA'
    else:
        string = 'Principal_Component_Analysis'
    print(string + ': '+str(convert_time(tnow - tlast)))
    tlast = tnow
    
    if feature == 'All':
        true_predictions = []
        false_predictions = []
        allscores = []
        for i in DICTIONARY:
            train_label, test_label = process_labels(attributes, number_samples, training_split, i, no_face_array)
            model, score = ML_Model(model_type, train_data, train_label, test_data, test_label)
            print(str(i)+' score = '+str(score))
            allscores.append(score)
        
        print('mean score = '+ str(np.mean(allscores)))

    else:
        
        train_label, test_label = process_labels(attributes, number_samples, training_split, feature, no_face_array)
        tnow = time.time()
        print('process_labels: '+str(convert_time(tnow - tlast)))
        tlast = tnow
        
        model, score = ML_Model(model_type, train_data, train_label, test_data, test_label)
        tnow = time.time()
        print('ML_Model: '+str(convert_time(tnow - tlast)))
        
        print('Model Accuracy: '+str(score))

    tsecs = time.time() - tstart
    string = convert_time(tsecs)
    print('Total Time Elapsed: '+str(string))

        