from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import numpy as np

app = FastAPI()

# load model
model = pickle.load(open("house_price.pkl", "rb"))

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request,
            income: float = Form(...),
            age: float = Form(...),
            rooms: float = Form(...),
            bedrooms: float = Form(...),
            population: float = Form(...)):

    features = np.array([[income, age, rooms, bedrooms, population]])

    prediction = model.predict(features)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": round(prediction[0], 2)
        }
    )