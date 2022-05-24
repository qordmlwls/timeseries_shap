import numpy as np
import tensorflow as tf
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from SPPModel_dasi import load_data, DataNotEnough, create_model, train

class ChannelTrain:

    def __init__(self, FEATURE_COLUMNS, N_STEPS, LOOKUP_STEP, TEST_SIZE, N_LAYERS, CELL, UNITS, DROPOUT, LOSS, OPTIMIZER, BATCH_SIZE, EPOCHS, path):
        self.FEATURE_COLUMNS = FEATURE_COLUMNS
        self.N_STEPS = N_STEPS
        self.LOOKUP_STEP = LOOKUP_STEP
        self.TEST_SIZE = TEST_SIZE
        self.N_LAYERS = N_LAYERS
        self.CELL = CELL
        self.UNITS = UNITS
        self.DROPOUT = DROPOUT
        self.LOSS = LOSS
        self.OPTIMIZER = OPTIMIZER
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.model = create_model(n_steps=self.N_STEPS, loss=self.LOSS, units=self.UNITS, cell=self.CELL, n_layers=self.N_LAYERS, dropout=self.DROPOUT,n_features=len(self.FEATURE_COLUMNS))
        self.path = path

    @staticmethod
    def check_gpu_and_update():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                [tf.config.experimental.set_memory_growth(g, True) for g in gpus]
            except RuntimeError as e:
                raise e
            except Exception as e:
                raise e
    def data_preparation(self, tmp_df):
        tmp_df['Date'] = tmp_df['Date'].astype(str)
        tmp_df = tmp_df.reset_index()
        tmp_df = tmp_df.fillna(method='backfill')
        tmp_df = tmp_df.fillna(method='ffill')
        tmp_df = tmp_df.fillna(tmp_df.mean())
        tmp_df = tmp_df[self.FEATURE_COLUMNS]
        tmp_df.rename(columns={'Close': 'close'}, inplace=True)
        return tmp_df

    def run(self, df):
        tmp_df = self.data_preparation(df)
        try:
            shuffled_data = load_data(df=tmp_df, n_steps=self.N_STEPS, lookup_step=self.LOOKUP_STEP, test_size=self.TEST_SIZE, shuffle=True)
        except DataNotEnough as e:
            print('Data is not enough')
            raise e
        model = self.model
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                shape = (layer.weights[0].shape[0], layer.weights[0].shape[1])
                y = 1.0 / np.sqrt(float(shape[0]))
                rule_weights = np.random.uniform(-y, y, shape)
                layer.weights[0] = rule_weights
                layer.weights[1] = 0
        history, stopped_epoch_list = train(shuffled_data, model, self.EPOCHS, self.BATCH_SIZE, verbose=1)
        # save model
        model.save_weights(os.path.join(self.path, 'model', 'model_for_predict_after%dweights.h5') %(self.LOOKUP_STEP))

# if __name__=='__main__':
#     start_time = timeit.default_timer()
#     ###############################Hyperparameter###############################################
#     """
#     needed to be set:
#     N_STEP: time_window for each sequence for training
#     LOOKUP_STEP: timedelta for prediction e.g. 7 or 30
#     """
#
#     class Arg:
#         FEATURE_COLUMNS = ["Open", "High", "Low", "Close"]
#         N_STEPS = 60
#         LOOKUP_STEP = 30
#         TEST_SIZE = 0.2
#         N_LAYERS = 4
#         CELL = GRU
#         UNITS = 50
#         DROPOUT = 0.2
#         LOSS = "mae"
#         OPTIMIZER = "adam"
#         BATCH_SIZE = 1024
#         EPOCHS = 130
#     ####################################load data##################################################
#     # distinguish whether run through debug or prompt
#     try:
#         stock_df = pd.read_csv(os.path.join('..', 'input', 'stock_data_example.csv'))
#         path = '..'
#     except FileNotFoundError:
#         path = os.getcwd()
#         stock_df = pd.read_csv(os.path.join(path, 'input', 'stock_data_example.csv'))
#     ###################################Training####################################################
#     args = Arg()
#     channel_predictioner = ChannelTrain(args.FEATURE_COLUMNS, args.N_STEPS, args.LOOKUP_STEP, args.TEST_SIZE,
#                                               args.N_LAYERS, args.CELL, args.UNITS, args.DROPOUT, args.LOSS,
#                                               args.OPTIMIZER, args.BATCH_SIZE, args.EPOCHS, path)
#     channel_predictioner.run(stock_df)