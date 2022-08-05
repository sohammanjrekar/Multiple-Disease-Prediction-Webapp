from django.contrib import admin
 
# add include to the path
from django.urls import path, include
 
# import views from todo
from ml import views
 
# import routers from the REST framework
# it is necessary for routing
from rest_framework import routers
 
# create a router object
router = routers.DefaultRouter()
 
# register the router
router.register(r'tasks',views.TodoView, 'task')
 
urlpatterns = [
    path('admin/', admin.site.urls),
 
    # add another path to the url patterns
    # when you visit the localhost:8000/api
    # you should be routed to the django Rest framework
    path('api/', include(router.urls))
 
 
]