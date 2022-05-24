# ml_timeseries_shap
- This project is for testing time series and shap
## Structure
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

## Shap output
![force](https://user-images.githubusercontent.com/43153661/169966279-76eac40d-e7cf-494e-938c-cbac3d4f9151.png)