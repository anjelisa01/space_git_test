# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# #berhasil konek dengan basic app di atas

#Berhasil konnek dgn script dibawah

from fastapi import FastAPI,Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import joblib
import numpy as np
from pydantic import BaseModel 
from typing import Optional

from feature_engineering import transform_to_model_ready_data  # Import function

model = joblib.load('the_model.joblib')
  
app=FastAPI()

class HouseFeatures(BaseModel):
  Id: int
  MSSubClass: int
  MSZoning: object
  LotFrontage: float
  LotArea: int
  Street: object
  Alley: object
  LotShape: object
  LandContour: object
  Utilities: object
  LotConfig: object
  LandSlope: object
  Neighborhood: object
  Condition1: object
  Condition2: object
  BldgType: object
  HouseStyle: object
  OverallQual: int
  OverallCond: int
  YearBuilt: int
  YearRemodAdd: int
  RoofStyle: object
  RoofMatl: object
  Exterior1st: object
  Exterior2nd: object
  MasVnrType: object
  MasVnrArea: float
  ExterQual: object
  ExterCond: object
  Foundation: object
  BsmtQual: object
  BsmtCond: object
  BsmtExposure: object
  BsmtFinType1: object
  BsmtFinSF1: float
  BsmtFinType2: object
  BsmtFinSF2: float
  BsmtUnfSF: float
  TotalBsmtSF: float
  Heating: object
  HeatingQC: object
  CentralAir: object
  Electrical: object
  firsttFlrSF: float
  secondFlrSF: float
  LowQualFinSF: float
  GrLivArea: float
  BsmtFullBath: int
  BsmtHalfBath: int
  FullBath: int
  HalfBath: int
  BedroomAbvGr: int
  KitchenAbvGr: int
  KitchenQual: object
  TotRmsAbvGrd: int
  Functional: object
  Fireplaces: int
  FireplaceQu: object
  GarageType: object
  GarageYrBlt: float
  GarageFinish: object
  GarageCars: int
  GarageArea: float
  GarageQual: object
  GarageCond: object
  PavedDrive: object
  WoodDeckSF: float
  OpenPorchSF: float
  EnclosedPorch: float
  threeSsnPorch: float
  ScreenPorch: float
  PoolArea: float
  PoolQC: object
  Fence: object
  MiscFeature: object
  MiscVal: float
  MoSold: int
  YrSold: int
  SaleType: object
  SaleCondition: object

# class MockModel:
#   def predict(self,X):
#     return['prediction_result']

# model=MockModel()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request:Request, exc:RequestValidationError):
  print("validation error:",exc.errors())
  return JSONResponse(status_code=422,content={"detail":exc.errors()})


@app.get('/')
def home(): 
  return {"message": "House price prediction is running!"}
@app.post('/predict/')
def predict_price(features:HouseFeatures):
  try:
    #model expecting json input
    input_data=np.array([[
        features.Id, features.MSSubClass, features.MSZoning, features.LotFrontage, features.LotArea,
        features.Street, features.Alley, features.LotShape, features.LandContour, features.Utilities,
        features.LotConfig, features.LandSlope, features.Neighborhood, features.Condition1, features.Condition2,
        features.BldgType, features.HouseStyle, features.OverallQual, features.OverallCond, features.YearBuilt,
        features.YearRemodAdd, features.RoofStyle, features.RoofMatl, features.Exterior1st, features.Exterior2nd,
        features.MasVnrType, features.MasVnrArea, features.ExterQual, features.ExterCond, features.Foundation,
        features.BsmtQual, features.BsmtCond, features.BsmtExposure, features.BsmtFinType1, features.BsmtFinSF1,
        features.BsmtFinType2, features.BsmtFinSF2, features.BsmtUnfSF, features.TotalBsmtSF, features.Heating,
        features.HeatingQC, features.CentralAir, features.Electrical, features.firsttFlrSF, features.secondFlrSF,
        features.LowQualFinSF, features.GrLivArea, features.BsmtFullBath, features.BsmtHalfBath, features.FullBath,
        features.HalfBath, features.BedroomAbvGr, features.KitchenAbvGr, features.KitchenQual, features.TotRmsAbvGrd,
        features.Functional, features.Fireplaces, features.FireplaceQu, features.GarageType, features.GarageYrBlt,
        features.GarageFinish, features.GarageCars, features.GarageArea, features.GarageQual, features.GarageCond,
        features.PavedDrive, features.WoodDeckSF, features.OpenPorchSF, features.EnclosedPorch, features.threeSsnPorch,
        features.ScreenPorch, features.PoolArea, features.PoolQC, features.Fence, features.MiscFeature, features.MiscVal,
        features.MoSold, features.YrSold, features.SaleType, features.SaleCondition
    ]])
    # Step 1: Transform raw data into model-ready data
    model_ready_data = transform_to_model_ready_data(input_data)
    # model_ready_data=model_ready_data.to_numpy()
    # model_ready_data=model_ready_data.reshape(1,-1)
    print("Transformed input:",model_ready_data)
    print("Input shape:", model_ready_data.shape)
      
    # Step 2: Use the model to predict
    prediction = model.predict(model_ready_data)  # Assuming model is already loaded
    return {"predicted_price":float(prediction[0])}
  except Exception as e:
    print("Erros:",str(e))
    return {"error":str(e)}




