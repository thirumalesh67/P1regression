from django.shortcuts import render
import joblib
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Create your views here.

def home_page(request):
    BASE_DIR = Path(__file__).resolve().parent.parent
    if request.method == 'POST':
        rf_model = joblib.load(os.path.join(BASE_DIR, 'models', 'rforest.pkl'))
        knn_model = joblib.load(os.path.join(BASE_DIR, 'models', 'knnModel.pkl'))
        gboost_model = joblib.load(os.path.join(BASE_DIR, 'models', 'gboostModel.pkl'))
        xgboost_model = joblib.load(os.path.join(BASE_DIR, 'models', 'xgboost.pkl'))
        stacking_model = joblib.load(os.path.join(BASE_DIR, 'models', 'stackingModel.pkl'))


        grlivarea = float(request.POST.get('GrLivArea'))
        lotarea = float(request.POST.get('LotArea'))
        totalbsmtsf = float(request.POST.get('TotalBsmtSF'))
        bsmtunfsf = float(request.POST.get('BsmtUnfSF'))
        garagearea = float(request.POST.get('GarageArea'))
        yearbuilt = int(request.POST.get('YearBuilt'))
        lotfrontage = float(request.POST.get('LotFrontage'))
        yearremodadd = int(request.POST.get('YearRemodAdd'))
        bsmtfinsf1 = float(request.POST.get('BsmtFinSF1'))
        openporchsf = float(request.POST.get('OpenPorchSF'))

        features = [grlivarea, lotarea, totalbsmtsf, bsmtunfsf, garagearea, yearbuilt, lotfrontage, yearremodadd,
                      bsmtfinsf1, openporchsf]
        rf_predict = rf_model.predict([features])
        knn_predict = knn_model.predict([features])
        gboost_predict = gboost_model.predict([features])
        xgboost_predict = xgboost_model.predict([features])

        predicted_price = stacking_model.predict([[rf_predict[0], knn_predict[0], gboost_predict[0], xgboost_predict[0]]])[0]

        context = {
            'grlivarea': grlivarea,
            'lotarea': lotarea,
            'totalbsmtsf': totalbsmtsf,
            'bsmtunfsf': bsmtunfsf,
            'garagearea': garagearea,
            'yearbuilt': yearbuilt,
            'lotfrontage': lotfrontage,
            'yearremodadd': yearremodadd,
            'bsmtfinsf1': bsmtfinsf1,
            'openporchsf': openporchsf,
            'predicted_price': predicted_price
        }
        return render(request, 'predict.html', context)
    return render(request, 'home.html')
