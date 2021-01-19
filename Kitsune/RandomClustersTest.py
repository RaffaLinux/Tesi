import numpy as np

def get_random_clusters(n_elem, n_clusters, random_state=0):
 k = int(np.floor(n_elem/n_clusters))
 np.random.seed(random_state)
 a=list(range(n_elem))
 c=[]
 for _ in range(n_clusters):
  c.append([])
  for _ in range(k):
   c[-1].append(a.pop(np.random.choice(list(range(len(a))))))
 i=-1
 for e in a:
  c[i].append(e)
  i-=1
 return c

for i in range(10):
    print(get_random_clusters(9,4,i))