import requests
import gpxpy
import gpxpy.gpx
import sys
import os
import numpy as np

print('Fetching Tracks')

# bbox = left,bottom,right,top
# bbox = min Longitude , min Latitude , max Longitude , max Latitude 

bbox = sys.argv[1:5]
directory = sys.argv[5]
print(bbox)
response = requests.get("https://api.openstreetmap.org/api/0.6/trackpoints?bbox={0},{1},{2},{3}&page=0".format(bbox[0],bbox[1],bbox[2],bbox[3]))

print(response)

gpx = gpxpy.parse(response.text)

try:
    os.mkdir(directory)
except:
    print("Could not make directory")

with open(f'{directory}/tracks_0.gpx','w') as file:
    file.write(gpx.to_xml())
    print('tracks_0.gpx')

count = 1

while True:
    response = requests.get("https://api.openstreetmap.org/api/0.6/trackpoints?bbox={0},{1},{2},{3}&page={4}".format(bbox[0],bbox[1],bbox[2],bbox[3],count))
    gpx = gpxpy.parse(response.text)
    if len(gpx.tracks) == 0:
        break
    with open('{0}/tracks_{1}.gpx'.format(directory,count),'w') as file:
        file.write(gpx.to_xml())
        print('Written to: tracks_{0}.gpx'.format(count))
    count = count + 1

print('Download Complete')


dataset = []

for file in os.listdir(directory):
    gpx_file = open(f'{directory}/{file}', 'r')
    gpx = gpxpy.parse(gpx_file)
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                dataset.append([point.longitude,point.latitude])



np.save(f"data/{directory}",dataset)

