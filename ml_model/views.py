from django.shortcuts import render
import pandas as pd
import joblib
import xgboost

model = joblib.load('./model/xgboostmodel1.pkl')
# Create your views here.
def home(request):
	return render(request,'home.html')

def predict(request):
	status = False
	print(request)
	if request.method == 'POST':
		dic={}
		dic['AGE']=request.POST.get('AGE')
		dic['STENOK_AN']=request.POST.get('STENOK_AN')
		dic['ZSN_AN']=request.POST.get('ZSN_AN')
		dic['S_AD_ORIT']=request.POST.get('S_AD_ORIT')
		dic['D_AD_ORIT']=request.POST.get('D_AD_ORIT')
		dic['K_SH_POST']=request.POST.get('K_SH_POST')
		dic['ant_im']=request.POST.get('ant_im')
		dic['ROE']=request.POST.get('ROE')
		dic['FIBR_JELUD']=request.POST.get('FIBR_JELUD')
		dic['RAZRIV']=request.POST.get('RAZRIV')
		status = True
		temp = dic.copy()
		print(dic.keys(),dic.values())

	df=pd.DataFrame({'input':dic}).transpose().astype("float")
	output = model.predict(df)[0]
	context = {'output':output,"temp":temp,"status":status}
	return render(request,'home.html',context)

 