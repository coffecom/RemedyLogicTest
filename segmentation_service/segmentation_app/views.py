from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .torch_model import predict


def upload_file(request):
    if request.method == 'POST':
        upload = request.FILES['file']
        
        # fss = FileSystemStorage()
        # file = fss.save(upload.name, upload) #так сохраняет, ура. 
    
        b64 = predict(upload.file)
        # print("B64:" + b64)
        return render(request, 'segmentation_success.html', context = {'prediction': b64})
    return render(request, 'upload.html')

def segmentation_success(request, prediction):
    return render(request, 'segmentation_success.html', {'prediction': prediction})
