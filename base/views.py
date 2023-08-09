from django.shortcuts import render , HttpResponse
import pandas as pd
import pickle

from sklearn.compose import ColumnTransformer


car = pd.read_csv('mlmodeldata/cleaned_car.csv')

model = pickle.load(open("mlmodeldata/LinearRegressionModel.pkl",'rb'))


def home(request):
      companies = sorted(car['company'].unique())
      name = sorted(car['name'].unique())
      year = sorted(car['year'].unique() , reverse=True)
      fuel_type = car['fuel_type'].unique()

      prediction = None
      if request.method == 'POST':
            company = request.POST['company']
            car_model = request.POST['car_model']
            years = request.POST['year']
            fueltype = request.POST['fueltype']
            kms_driven = request.POST['kms_driven']
            print(company)
            prediction = model.predict(pd.DataFrame([[company , car_model , year ,kms_driven , fueltype ]] ,columns=['name' , 'company' , 'year'  , 'kms_driven' , 'fuel_type' ]))

      # pred = prediction
      context = {'companies':companies , 'name':name , 'year':year , 'fuel_type':fuel_type }
      return render(request , 'index.html' ,context)