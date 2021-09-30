# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 19:17:59 2020

@author: Zikantika
"""
import math

# Fractional number.
n = 100.7

# Absolute value.
print(math.floor(n))
print(math.ceil(n))

mass_kg = int(input("What is your mass in kilograms?" ))
mass_stone = mass_kg * 2.2 / 14
print("You weigh", mass_stone, "stone.")



def mean(data_list):
	data_list = map(float, data_list)
	return sum(data_list) / len(data_list)


import math

# Input list.
values = [0.9999999, 1, 2, 3]

# Sum values in list.
r = sum(values)
print(r)

# Sum values with fsum.
r = math.fsum(values)
print(r)


import math

# Truncate this value.
value1 = 123.45
truncate1 = math.trunc(value1)
print(truncate1)

# Truncate another value.
value2 = 345.67
truncate2 = math.trunc(value2)
print(truncate2)