from django.shortcuts import render
import joblib

# Create your views here.

def home_page(request):
    if request.method == 'POST':
        model = joblib.load('../../models/rforest.pkl')

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

        input_data = [grlivarea, lotarea, totalbsmtsf, bsmtunfsf, garagearea, yearbuilt, lotfrontage, yearremodadd,
                      bsmtfinsf1, openporchsf]
        predicted_price = model.predict([input_data])[0]
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
