import os
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET

class Control(object):

    def __init__(self):
        #self.summaries = tf.summary.merge_all()
        #self.feed_dict = {}
        self.average_travel_times = []
        #self.graph = tf.Graph() 
        dirname = os.path.dirname(__file__)
        #self.create_placeholder()
        self.output = os.path.join(dirname, 'Stats')
        if os.path.isdir(self.output):
            print("output_results: ", str(self.output))
        else:
            os.mkdir(self.output)
        #self.writer = tf.summary.FileWriter(self.output, self.graph)
        #self.summaries = tf.summary.merge_all()
           
    '''def create_placeholder(self):
        self.info_vars = ['total_waiting', 'total_delay', 'result', 'num_teleports', 'emergency_stops']
        with self.graph.as_default():
            for name in self.info_vars:
                variable = tf.placeholder(tf.float32, [None,], name=name)
                tf.summary.scalar(variable.name, tf.reduce_mean(variable))

            variable = tf.placeholder(tf.int32, [], name='travel_time')
            tf.summary.scalar('Average_TT', variable)'''

    '''def intialise_dict(self):
        with self.graph.as_default():
            for name in self.info_vars:
                key = tf.get_default_graph().get_tensor_by_name(name+':0')
                self.feed_dict[key] = []'''

    '''def add_reward(self, reward):
        with self.graph.as_default():
            for name in self.info_vars:
                try:
                    key = tf.get_default_graph().get_tensor_by_name(name+':0')
                    self.feed_dict[key].append(reward[name])
                except:
                    continue'''

    def log(self):
        """
        Calculate the mean of all the values we store per factor and write
        the summaries.
        """
        # Read the output file of SUMO to retrieve the travel times of the cars
        success = False
   
        for files in os.listdir(self.output):
            if files.endswith(".xml"):
               trip_info_file = files
               break
        
        self.trip_info_file = os.path.join(self.output, trip_info_file)

        while not success:
            # Try to copy the output file of SUMO
            try:
                tree = ET.parse(self.trip_info_file)
                success = True
            except:
                print("Could not load output file.")
                success = False
                
        data = tree.getroot()
        total_travel_time = []
        
        for car in data:
            total_travel_time.append(float(car.attrib['duration']))
        
        '''with self.graph.as_default():
            key = tf.get_default_graph().get_tensor_by_name('travel_time:0')'''
        average_travel_time = sum(total_travel_time)/len(total_travel_time)
            #self.feed_dict[key] = average_travel_time
        self.average_travel_times.append(average_travel_time)
        
        return self.average_travel_times, average_travel_time
        
        '''with tf.Session(graph=self.graph) as sess:
            summary = sess.run(self.summaries, feed_dict=self.feed_dict)
            self.writer.add_summary(summary, global_step)'''
      
