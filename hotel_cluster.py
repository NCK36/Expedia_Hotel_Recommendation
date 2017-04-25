# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:22:32 2017

@author: Navin Chandan
"""

from heapq import nlargest
from operator import itemgetter
from collections import defaultdict

#Checking columns name and creating empty dictionaries 
f = open("train.csv", "r")
f.readline()
ulc_odd_dict = defaultdict(lambda: defaultdict(int))   #corresponding user location city and origin destination distance
srch_dest_dict1 = defaultdict(lambda: defaultdict(int))
srch_dest_dict2 = defaultdict(lambda: defaultdict(int))
hotel_country_dict = defaultdict(lambda: defaultdict(int))
popular_hotel_cluster = defaultdict(int)

#Initialising total to read limited number of lines
total = 0

#Reading train dataset line by line and fetching relevant data
while 1:
    line = f.readline().strip()
    total += 1

    if total == 10000000:
        break
        
    arr = line.split(",")
    book_year = int(arr[0][:4])
    user_location_city = arr[5]
    orig_destination_distance = arr[6]
    srch_destination_id = int(arr[16])
    is_booking = int(arr[18])
    hotel_country = arr[21]
    hotel_market = arr[22]
    hotel_cluster = arr[23]

    w1 = 3 + 17*is_booking   #More weightage to booking than to click
    w2 = 1 + 5*is_booking

# storing hotel cluster corresponding to user location city and origin destination distance
    if user_location_city != '' and orig_destination_distance != '':
        ulc_odd_dict[(user_location_city, orig_destination_distance)][hotel_cluster] += 1

# storing hotel cluster corresponding to search destination id, hotel country and hotel market
    if srch_destination_id != '' and hotel_country != '' and hotel_market != '' and book_year == 2014:
        srch_dest_dict1[(srch_destination_id, hotel_country, hotel_market)][hotel_cluster] += w1

    if srch_destination_id != '':
        srch_dest_dict2[srch_destination_id][hotel_cluster] += w1
        
    if hotel_country != '':
        hotel_country_dict[hotel_country][hotel_cluster] += w2

# storing popular hotel cluster
    popular_hotel_cluster[hotel_cluster] += 1

f.close()

#creating the submission file
path = 'submission' + '.csv'
out = open(path, "w")

#checking columns name for test dataset
f = open("test.csv", "r")
f.readline()

#column names for submission file
out.write("id,hotel_cluster\n")

#Sorting the popular hotel cluster and taking out top five 
topclasters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

#Reading and predicting hotel cluster on test data
while 1:
    line = f.readline().strip()

    if line == '':
        break

    arr = line.split(",")
    id = arr[0]
    user_location_city = arr[6]
    orig_destination_distance = arr[7]
    srch_destination_id = arr[17]
    hotel_country = arr[20]
    hotel_market = arr[21]

    out.write(str(id) + ',')
    filled = []

# Checking if user location city and origin destination dist from test dataset is in
# dictionary created based on train dataset
    s1 = (user_location_city, orig_destination_distance)
    if s1 in ulc_odd_dict:
        d = ulc_odd_dict[s1]
        topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
        for i in range(len(topitems)):
            if topitems[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            out.write(' ' + topitems[i][0])
            filled.append(topitems[i][0])

    s2 = (srch_destination_id, hotel_country, hotel_market)
    if s2 in srch_dest_dict1:
        d = srch_dest_dict1[s2]
        topitems = nlargest(5, d.items(), key=itemgetter(1))
        for i in range(len(topitems)):
            if topitems[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            out.write(' ' + topitems[i][0])
            filled.append(topitems[i][0])
    elif srch_destination_id in srch_dest_dict2:
        d = srch_dest_dict2[srch_destination_id]
        topitems = nlargest(5, d.items(), key=itemgetter(1))
        for i in range(len(topitems)):
            if topitems[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            out.write(' ' + topitems[i][0])
            filled.append(topitems[i][0])
        
    if hotel_country in hotel_country_dict:
        d = hotel_country_dict[hotel_country]
        topitems = nlargest(5, d.items(), key=itemgetter(1))
        for i in range(len(topitems)):
            if topitems[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            out.write(' ' + topitems[i][0])
            filled.append(topitems[i][0])

#if any id is left black it is filled with topcluster
    for i in range(len(topclasters)):
        if topclasters[i][0] in filled:
            continue
        if len(filled) == 5:
            break
        out.write(' ' + topclasters[i][0])
        filled.append(topclasters[i][0])

    out.write("\n")
out.close()
