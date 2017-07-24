import numpy as np
import pandas as pd

def generate_data(data_size=100000):
    """
     This function generates a dataset for CFME deployment.
    
    """

    # INPUT FEATURES

    #number of objects to collect data from
    num_objects_to_collect = np.random.randint(1000, 30000, size=data_size)

    #Metrics collection time
    metrics_collection_time = np.random.uniform(1, 150, size=data_size)

    #estimated PSS memory per N Objects
    estimated_pss_memory = (np.random.rand(data_size)*100).astype(np.int) + 1

    #base worker pss memory
    base_worker_memory = np.random.choice([250, 500, 750, 1000, 
                                       1250, 1500, 2000, 2500],
                                      size=data_size,
                                      p=[0.05, 0.35, 0.25, 0.2,
                                        0.05, 0.05, 0.025, 0.025]
                                     )

    #maximum pss memory per worker
    maximum_pss_memory = np.random.choice([2, 3, 4], size=data_size) * base_worker_memory

    #percent cpu utilization 
    percent_cpu_util = np.random.rand(data_size)

    #OUTPUT LABELS

    # number of workers running the workload
    num_workers = np.random.randint(4, 100, size=data_size)

    # memory threshold of each of the workers
    worker_memory_threshold = np.random.choice([250, 500],size=data_size, p=[0.2, 0.8])

    # Create a dataframe for training the models to tune CFME
    train_features = {"num_objects_to_collect": num_objects_to_collect,
                  "metrics_collection_time": metrics_collection_time,
                  "estimated_pss_memory": estimated_pss_memory,
                  "base_worker_memory": base_worker_memory,
                  "maximum_pss_memory": maximum_pss_memory,
                  "percent_cpu_util": percent_cpu_util,
                  "workers_num": num_workers,
                  "worker_memory_threshold": worker_memory_threshold
                 }

    train = pd.DataFrame(train_features)

    return train
