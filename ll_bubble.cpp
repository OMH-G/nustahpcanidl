#include<iostream>
#include<omp.h>
using namespace std;


void oddEvenBubble(int arr[],int size){
bool sorted = false;

while(!sorted){

  sorted = true;

  #pragma omp parallel shared(arr,size,sorted)
  {
    #pragma omp for schedule(static)
    for(int i=0;i<size-1;i+=2){
      if(arr[i]>arr[i+1]){
        std::swap(arr[i],arr[i+1]);
        sorted=false;
      }
    }

    #pragma omp for schedule(static)
      for(int j=1;j<size-1;j+=2){
        if(arr[j]>arr[j+1]){
          std::swap(arr[j],arr[j+1]);
          sorted = false;

        }

      }

    }
}

}


int main(){

  int arr[] = {6,3,0,5};
  int size = 4;

oddEvenBubble(arr,size);

for(int i=0;i<size;i++){
cout<<arr[i];

}

return 0;
}
