from django import forms 
from .models import ImageEntity

class ImageEntityForm(forms.ModelForm): 
    class Meta: 
        model = ImageEntity 
        fields = ['image'] 