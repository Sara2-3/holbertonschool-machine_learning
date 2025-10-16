#!/usr/bin/env python3
"""This is first task with name Slice Me Up."""

arr = [9, 8, 2, 3, 9, 4, 1, 0, 3]

# per te bere slice perdoret [start:end], numerimi fillon nga 0, 1, 2
arr1 = arr[0:2]  # dy elementët e parë

# arr[4:9] edhe ketu funksion, por me indeks negativ është më mirë
arr2 = arr[-5:]  # pesë elementët e fundit

# nuk është [2:7], kërkohen elementët jo indekset
# pra nisesh nga numerim normal
arr3 = arr[1:6]

print("The first two numbers of the array are: {}".format(arr1))
print("The last five numbers of the array are: {}".format(arr2))
print("The 2nd through 6th numbers of the array are: {}".format(arr3))
