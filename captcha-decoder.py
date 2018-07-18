# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 19:52:12 2018

@author: zas
"""

from __future__ import absolute_import
import os
import logging
import base64
import argparse
from code.functions.helpers import predict_captcha_text


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', action='store', dest='captcha', help='Provide the captcha')
    parser.add_argument('-t', action='store', dest='type', default='url', help='Captcha type [local file/ url / base64]')
    parser.add_argument('-m', action='store', dest='model', default='keras', help='Solver model [Keras/NN]')
    
    results = parser.parse_args()


    image_path = results.captcha
    img_type = results.type
    model = results.nodel


    image_path2 =image_path.replace(' ', '+')
    captcha_text = predict_captcha_text(str(image_path2),img_type,model)
    
    print(captcha_text)
