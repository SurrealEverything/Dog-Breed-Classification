#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:58:26 2018

@author: gabriel
"""
import random

def get_binary_gender(d, name):
	gdr = d.get_gender(name) 
	
	if(gdr == 'male'):
		bnr_gdr = 0
	elif(gdr == 'mostly_male'):
		bnr_gdr = 0
	elif(gdr == 'female'):
		bnr_gdr = 1
	elif(gdr == 'mostly_female'):
		bnr_gdr = 1
	else:
		bnr_gdr = random.uniform(0, 1)
	
	return bnr_gdr



