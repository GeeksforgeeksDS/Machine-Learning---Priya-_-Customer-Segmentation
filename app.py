from fastapi import FastAPI, Request
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from uvicorn import run as app_run
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.constant.application import *

import warnings
warnings.filterwarnings('ignore')

app = FastAPI()


templates = Jinja2Templates(directory='templates')


origins = ["*"]

app.mount("/static", StaticFiles(directory="static"), name="static")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.Age : Optional[str] = None
        self.Education  : Optional[str] = None
        self.Marital_Status  : Optional[str] = None
        self.Parental_Status : Optional[str] = None
        self.Children  : Optional[str] = None
        self.Income  : Optional[str] = None
        self.Total_Spending  : Optional[str] = None
        self.Days_as_Customer  : Optional[str] = None
        self.Recency  : Optional[str] = None
        self.Wines  : Optional[str] = None
        self.Fruits  : Optional[str] = None
        self.Meat : Optional[str] = None
        self.Fish   : Optional[str] = None
        self.Sweets : Optional[str] = None
        self.Gold  : Optional[str] = None
        self.Web  : Optional[str] = None
        self.Catalog  : Optional[str] = None
        self.Store  : Optional[str] = None
        self.Discount_Purchases  : Optional[str] = None
        self.Total_Promo  : Optional[str] = None
        self.NumWebVisitsMonth  : Optional[str] = None
        

    async def get_customer_data(self):
        form =  await self.request.form()
        self.Age = form.get('Age')
        self.Education = form.get('Education')
        self.Marital_Status = form.get('Marital_Status')
        self.Parental_Status = form.get('Parental_Status')
        self.Children = form.get('Children')
        self.Income = form.get('Income')
        self.Total_Spending = form.get('Total_Spending')
        self.Days_as_Customer = form.get('Days_as_Customer')
        self.Recency = form.get('Recency')
        self.Wines = form.get('Wines')
        self.Fruits = form.get('Fruits')
        self.Meat = form.get('Meat')
        self.Fish = form.get('Fish')
        self.Sweets = form.get('Sweets')
        self.Gold = form.get('Gold')
        self.Web = form.get('Web')
        self.Catalog = form.get('Catalog')
        self.Store = form.get('Store')
        self.Discount_Purchases = form.get('Discount_Purchases')
        self.Total_Promo = form.get('Total_Promo')
        self.NumWebVisitsMonth = form.get('NumWebVisitsMonth')

@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.get("/")
async def predictGetRouteClient(request: Request):
    try:

        return templates.TemplateResponse(
            "customer.html",
            {"request": request, "context": "Rendering"},
        )

    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        
        await form.get_customer_data()
        
        input_data = [form.Age, 
                    form.Education, 
                    form.Marital_Status, 
                    form.Parental_Status, 
                    form.Children, 
                    form.Income, 
                    form.Total_Spending, 
                    form.Days_as_Customer, 
                    form.Recency, 
                    form.Wines, 
                    form.Fruits, 
                    form.Meat, 
                    form.Fish, 
                    form.Sweets, 
                    form.Gold, 
                    form.Web, 
                    form.Catalog, 
                    form.Store, 
                    form.Discount_Purchases, 
                    form.Total_Promo, 
                    form.NumWebVisitsMonth]
        
        prediction_pipeline = PredictionPipeline()
        predicted_cluster = prediction_pipeline.run_pipeline(input_data=input_data)
       
        
        # model_predictor = Customer_segmentation_Classifier()

        # predicted_cluster = model_predictor.predict(customer_data_df)
        return templates.TemplateResponse(
            "customer.html",
            {"request": request, "context": int(predicted_cluster[0])}
        )

    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host = APP_HOST, port =APP_PORT)
    
