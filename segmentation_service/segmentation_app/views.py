from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage


def upload_file(request):
    if request.method == 'POST':
        upload = request.FILES['file']
        fss = FileSystemStorage()
        file = fss.save(upload.name, upload) #так сохраняет, ура. 
        return HttpResponseRedirect('/segmentation/segmentation_success')  
    return render(request, 'upload.html')

def segmentation_success(request):
    images = [[1,1,1]]
    return render(request, 'segmentation_success.html', {'images': images})
