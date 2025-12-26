# Pandas

Notes for Tasks
# 4.Task
.values është një atribut i pandas DataFrame që kthen automatikisht numpy array pa nevojë për import të numpy

# 5.Task
[::60] slicing step start from 0 move with 60,
.iat[] this is betten than  .iloc[], More efficient for individual value 
# 6.Task
return df.sort_values('Timestamp', ascending=False).T
.T make transpose
# 7.Task
subset=['Close'] specifikon që të hiqen vetëm rreshtat ku kolona 'Close' ka NaN
Mban të gjitha kolonat e tjera
Kthen të gjithë DataFrame-in