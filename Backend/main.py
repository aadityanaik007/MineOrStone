import FastAPI
import datetime
from ML.ml import Trainer_class
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    print("Server Started on ",datetime.datetime.now())
    Trainer_class.build_model()

@app.post('/get_prediction')
async def get_prediction(inputdata:tuple):
   response = Trainer_class.get_prediction(inputdata)
   return response