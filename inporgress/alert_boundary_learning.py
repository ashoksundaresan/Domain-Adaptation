import numpy as np
from sklearn.neighbors import DistanceMetric
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
def extract_data_from_log(log_file_path,all_images,disagreed_images):
    data=pd.read_csv(log_file_path,header=None)
    queried_data = data[data[1].isin(all_images)]
    disagreed_data = data[data[1].isin(disagreed_images)]
    return  queried_data, disagreed_data

def extract_prob_matrix(df,col,obj_cols=['Person','TV','Chair'],num_items_per_col=3):
    prob_data=df[col]
    prob_matrix=np.zeros((df.shape[0],len(obj_cols)*num_items_per_col))
    print prob_matrix.shape
    for probs in prob_data:
        print probs
        break



if __name__=='__main__':
    log_file_path='/Users/karthikkappaganthu/Documents/online_learning/data/log_master.csv'
    all_images=['/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003625952.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003626468.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003627309.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003628165.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003629301.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003630694.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003643236.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003645274.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003645863.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003647232.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003647871.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003648777.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003649331.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003650311.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003651674.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003652450.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003653321.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003654476.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003655224.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003656237.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003657912.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003658382.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003659346.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003700207.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003701431.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003702406.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003703402.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003707833.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003708358.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003709750.jpg']

    disagreed_images=['/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003647232.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003647871.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003648777.jpg',
'/var/www/html/server/data/Tyco Innovation Garage/Z/existing video/2017-07-28/00/20170728-003649331.jpg']

    queried_data, disagreed_data=extract_data_from_log(log_file_path, all_images, disagreed_images)
    print queried_data.shape
    print disagreed_data.shape

    extract_prob_matrix(queried_data,3)







