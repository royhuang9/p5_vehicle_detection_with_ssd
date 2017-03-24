import numpy as np
import os

width = 1242
height = 375
# we just care about vehicle and pedistrain
class data_preprocessor(object):

    def __init__(self, data_path):
        self.path_prefix = data_path
        self.num_classes = 2
        self.data = dict()
        self._preprocess_XML()

    def _preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)
        for filename in filenames:
            full_dir = os.path.join(self.path_prefix, filename)
            #print(full_dir)
            with open(full_dir) as fp:
                for l in fp:
                    data = l.split()
                    bboxes = []
                    one_hot_classes = []
                    veh_or_ped, one_hot_class = self._to_one_hot(data[0])
                    if veh_or_ped:
                        xmin = float(data[4])/width
                        ymin = float(data[5])/height
                        xmax = float(data[6])/width
                        ymax = float(data[7])/height
                        bboxes.append((xmin, ymin, xmax, ymax))
    
                    one_hot_classes.append(one_hot_class)
            if bboxes:
                image_name = os.path.split(filename)[0]
                bboxes = np.asarray(bboxes)
                one_hot_classes = np.asarray(one_hot_classes)
                image_data = np.hstack((bboxes, one_hot_classes))
                self.data[image_name] = image_data

    def _to_one_hot(self,name):
        one_hot_vector = [0] * self.num_classes
        veh_or_pred = False
        if name == 'Car':
            one_hot_vector[0] = 1
            veh_or_pred = True
        elif name == 'Van':
            one_hot_vector[0] = 1
            veh_or_pred = True
        elif name == 'Truck':
            one_hot_vector[0] = 1
            veh_or_pred = True
        elif name == 'Cyclist':
            one_hot_vector[0] = 1
            veh_or_pred = True
        elif name == 'Pedestrian':
            one_hot_vector[1] = 1
            veh_or_pred = True
        elif name == 'Person_sitting':
            one_hot_vector[1] = 1
            veh_or_pred = True
        elif name == 'Tram':
            one_hot_vector[1] = 1
            veh_or_pred = True
        else:
            #print('unknown label: %s' %name)
            pass

        return veh_or_pred, one_hot_vector

# example on how to use it
import pickle
data = data_preprocessor('/home/roy/data/kitti/training/label_2').data
pickle.dump(data,open('kitti_training.pkl','wb'))

