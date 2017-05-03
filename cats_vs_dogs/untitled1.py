# -*- coding: utf-8 -*-
"""
Created on Wed May  3 10:42:05 2017

@author: Max
"""

import os

path = 'C:/Users/Max/Documents/tf_test/test'
z = '.jpg'

for file in os.listdir(path):
    if file.find('.')<0:
        newname=file+'.jpg'
        os.rename(os.path.join(path,file),os.path.join(path,newname))
        print(file,'ok')
    