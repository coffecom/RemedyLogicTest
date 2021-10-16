from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.shortcuts import render

def handle_uploaded_file(file):
    print("HELLO")


def upload_file(request):
    if request.method == 'POST':
        if  len(request.FILES) > 0:
            handle_uploaded_file(request.FILES['file'])
            return HttpResponseRedirect('/segmentation/segmentation_success')
        else: print("NO FILES")
    # else:
    return render(request, 'upload.html')

def segmentation_success(request):
    images = [[1,1,1]]
    return render(request, 'segmentation_success.html', {'images': images})
