from django.shortcuts import render,redirect

# Create your views here.
from django.http import HttpResponse
from django.views import View
from .mask_rcnn_detection import detect_and_compare
from .forms import ImageUploadForm
from .models import ImageUpload


class Disease(View):

    def get(self, request):
        detect_and_compare()
        
        return render(request, 'index.html',)
    
    def post(self,request):
        upload_image(request)
        
def upload_image(request):
        if request.method == 'POST':
            form = ImageUploadForm(request.POST, request.FILES)
            if form.is_valid():
                form.save()
                
                # detect_and_compare(image_upload.image.path)
                return redirect('success')
        else:
            form = ImageUploadForm()
        # return render(request, 'upload.html', {'form': form})