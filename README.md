# ml_timeseries_shap

## 구조
- [input]
  - stock_data_example.csv (모델 학습, prediction 데이터)
  
- [output]
  - force.png (XAI 분석 결과 force plot)
  
- [model] 
  - 학습된 모델 저장

- [modules]
  - prediction_shap.py -> prediction과 XAI 분석 수행 후 forceplot 저장
  - sns_ch_train.py -> 모델 학습
  - SPPModel_dasi.py -> 모델 학습시 필요 모듈 

- [test] : 모듈 실행 파일