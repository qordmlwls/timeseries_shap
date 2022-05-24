import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from modules.sns_ch_train import ChannelTrain
import pandas as pd
from tensorflow.keras.layers import GRU
import timeit




if __name__=='__main__':
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
    ####################################load data##################################################
    # distinguish whether run through debug or prompt
    try:
        stock_df = pd.read_csv(os.path.join('..', 'input', 'stock_data_example.csv'))
        path = '..'
    except FileNotFoundError:
        path = os.getcwd()
        stock_df = pd.read_csv(os.path.join(path, 'input', 'stock_data_example.csv'))
    ###################################Training####################################################
    args = Arg()
    channel_predictioner = ChannelTrain(args.FEATURE_COLUMNS, args.N_STEPS, args.LOOKUP_STEP, args.TEST_SIZE,
                                              args.N_LAYERS, args.CELL, args.UNITS, args.DROPOUT, args.LOSS,
                                              args.OPTIMIZER, args.BATCH_SIZE, args.EPOCHS, path)
    channel_predictioner.run(stock_df)