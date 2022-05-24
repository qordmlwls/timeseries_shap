# ml_timeseries_shap

## 구조
- [input]
  - stock_data_example.csv (training, prediction data)
  
- [output]
  - force.png (XAI analysis force plot)
  
- [model] 
  - save trained model

- [modules]
  - prediction_shap.py -> predict and save forceplot 
  - sns_ch_train.py -> train model
  - SPPModel_dasi.py -> module for train model

- [test] : excute train, prediction and XAI