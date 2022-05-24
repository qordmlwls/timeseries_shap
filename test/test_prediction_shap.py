import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from tensorflow.keras.layers import GRU
import timeit
from modules.prediction_shap import ChannelPredictionShap
import tensorflow as tf
# shap is not compatible with tensorflow > 2.0
tf.compat.v1.disable_v2_behavior()

if __name__ == '__main__':
    start_time = timeit.default_timer()
    ###############################Hyperparameter###############################################
    """
    needed to be set: 
    N_STEP: time_window for each sequence for training
    LOOKUP_STEP: timedelta for prediction e.g. 7 or 30 
    """

    class Arg:
        FEATURE_COLUMNS = ["Open", "High", "Low", "Close"]
        N_STEPS = 60
        LOOKUP_STEP = 30
        TEST_SIZE = 0.2
        N_LAYERS = 4
        CELL = GRU
        UNITS = 50
        DROPOUT = 0.2
        LOSS = "mae"
        OPTIMIZER = "adam"
        BATCH_SIZE = 1024
        EPOCHS = 130
    try:
        df = pd.read_csv(os.path.join('..', 'input', 'stock_data_example.csv'))
        path = '..'
    except FileNotFoundError:
        df = pd.read_csv(os.path.join(os.getcwd(), 'input', 'stock_data_example.csv'))
        path = os.getcwd()

    args = Arg()
    shap_prediction = ChannelPredictionShap(args.FEATURE_COLUMNS, args.N_STEPS, args.LOOKUP_STEP, args.TEST_SIZE,
                                              args.N_LAYERS, args.CELL, args.UNITS, args.DROPOUT, args.LOSS,
                                              args.OPTIMIZER, args.BATCH_SIZE, args.EPOCHS, path)
    shap_prediction.run(df)