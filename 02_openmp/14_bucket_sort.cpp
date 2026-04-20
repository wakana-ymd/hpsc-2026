#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 
  for (int i=0; i<n; i++)
    bucket[key[i]]++;
  std::vector<int> offset(range,0);
  for (int i=1; i<range; i++) 
    offset[i] = offset[i-1] + bucket[i-1];

  #pragma omp parallel for shared(bucket,offset,key)
  for (int i=0; i<range; i++) {
    int j = offset[i];
    int cnt = bucket[i];
    for (int k=0; k<cnt; k++) {
      key[j+k] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
