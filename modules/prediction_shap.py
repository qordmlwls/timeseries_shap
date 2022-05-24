import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from SPPModel_dasi import load_data, create_model
import shap

class ChannelPredictionShap:
    def __init__(self,FEATURE_COLUMNS,N_STEPS,LOOKUP_STEP,TEST_SIZE,N_LAYERS,CELL,UNITS,DROPOUT,LOSS,OPTIMIZER,BATCH_SIZE,EPOCHS, path):
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

    def data_preparation(self, new_df):
        new_df['Date'] = new_df['Date'].astype(str)
        new_df = new_df.reset_index()
        new_df = new_df.fillna(method='backfill')
        new_df = new_df.fillna(method='ffill')
        new_df = new_df.fillna(new_df.mean())
        new_df = new_df[self.FEATURE_COLUMNS]
        new_df.rename(columns={'Close': 'close'}, inplace=True)
        return new_df
    def load_model(self):
        model = self.model
        model.load_weights(os.path.join(self.path, 'model', 'model_for_predict_after%dweights.h5') %(self.LOOKUP_STEP))
        return model
    def run(self, df):
        new_df = self.data_preparation(df)
        model = self.load_model()
        data = load_data(df=new_df, n_steps=self.N_STEPS, lookup_step=self.LOOKUP_STEP, test_size=self.TEST_SIZE,
                         shuffle=False)
        y_test = data["y_test"]
        X_test = data["X_test"]

        explainer = shap.DeepExplainer(model, data=X_test)
        shap_values = explainer.shap_values(X_test)
        shap.force_plot(explainer.expected_value, shap_values[0][0, 0, :], X_test[0, 0, :], feature_names=['Open', 'High', 'Low', 'Close'] ,show=False, matplotlib=True).savefig(self.path + '\\output\\force.png') # importance plot

