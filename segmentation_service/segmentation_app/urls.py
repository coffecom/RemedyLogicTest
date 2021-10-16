from django.urls import path

from . import views

urlpatterns = [
    path('load/', views.upload_file, name = 'load'),
    path('segmentation_success/', views.segmentation_success, name='segmentation_success')
]