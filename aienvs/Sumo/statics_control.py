import os
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET

class Control(object):

    def __init__(self, tripinfofolder=None):
        
        self.tripinfofolder = tripinfofolder
        self.average_travel_times = []
        self.dirname = '/home/azlaan/PycharmProjects/otherprojects/aienvs/test/Stats/' + str(self.tripinfofolder)
        
    def log(self):
        """
        Calculate the mean of all the values we store per factor and write
        the summaries.
        """
        # Read the output file of SUMO to retrieve the travel times of the cars
        success = False
        i = 0 
   
        for files in os.listdir(self.dirname):
            if files.endswith(".xml"):
               trip_info_file = files
               break
        self.trip_info_file = os.path.join(self.dirname, trip_info_file)

        while not success:
            # Try to copy the output file of SUMO
            try:
                tree = ET.parse(self.trip_info_file)
                success = True
            except:
                print("Could not load output file.")
                success = False

                i +=1
                if i == 20:
                    continue
                
        data = tree.getroot()
        total_travel_time = []
        avg_speed = []
        
        for car in data:
            total_travel_time.append(float(car.attrib['duration']))
            avg_speed.append(float(car.attrib['routeLength'])/float(car.attrib['duration']))
        
        average_speed = sum(avg_speed)/len(avg_speed)
        average_travel_time = sum(total_travel_time)/len(total_travel_time)
        self.average_travel_times.append(average_travel_time)
        
        return self.average_travel_times, average_travel_time, average_speed
