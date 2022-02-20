import pandas as pd
from tqdm import tqdm
import numpy as np

'''
df = pd.read_csv('../data/reformat3t_without_imputation.csv')

for k in ['FiO2_1', 'PEEP', 'TidalVolume']:
    values = df[k].values
    values[np.isnan(values)] = np.nanmean(values)
    print(k, values.min(), values.max())

icustayid_selected = set()
for icustayid in tqdm(df['icustayid'].unique()):
    data = df[df['icustayid'] == icustayid]
    if data['mechvent'].values.sum() > 6:
        icustayid_selected.add(icustayid)
print('There are {:d} patients'.format(len(icustayid_selected)))

'''

mech_settings = ['FiO2_1', 'PEEP', 'TidalVolume']
df = pd.read_csv('../data/mimictable_mech.csv')
# df[df['mechvent']==0][mech_settings] = 0


reformat4t = df
s=reformat4t.loc[:,['PaO2_FiO2','Platelets_count','Total_bili','MeanBP','max_dose_vaso','GCS','Creatinine','output_4hourly']].values  
p=np.array([0, 1, 2, 3, 4]) 

s1=np.transpose(np.array([s[:,0]>400, (s[:,0]>=300) & (s[:,0]<400), (s[:,0]>=200) & (s[:,0]<300), (s[:,0]>=100) & (s[:,0]<200), s[:,0]<100 ]))  #count of points for all 6 criteria of sofa
s2=np.transpose(np.array([s[:,1]>150, (s[:,1]>=100) & (s[:,1]<150), (s[:,1]>=50) & (s[:,1]<100), (s[:,1]>=20) & (s[:,1]<50), s[:,1]<20 ]))
s3=np.transpose(np.array([s[:,2]<1.2, (s[:,2]>=1.2) & (s[:,2]<2), (s[:,2]>=2) &(s[:,2]<6), (s[:,2]>=6) & (s[:,2]<12), s[:,2]>12 ]))
s4=np.transpose(np.array([s[:,3]>=70, (s[:,3]<70) & (s[:,3]>=65), s[:,3]<65, (s[:,4]>0) & (s[:,4]<=0.1), s[:,4]>0.1 ]))
s5=np.transpose(np.array([s[:,5]>14, (s[:,5]>12) & (s[:,5]<=14), (s[:,5]>9) & (s[:,5]<=12), (s[:,5]>5) & (s[:,5]<=9), s[:,5]<=5 ]))
s6=np.transpose(np.array([s[:,6]<1.2, (s[:,6]>=1.2) & (s[:,6]<2), (s[:,6]>=2) & (s[:,6]<3.5), ((s[:,6]>=3.5) & (s[:,6]<5)) | (s[:,7]<84), (s[:,6]>5) | (s[:,7]<34) ]))


reformat4=np.zeros((len(reformat4t), 7))


for i in tqdm(range(0,reformat4.shape[0])):
    for j, sx in enumerate([s1, s2, s3, s4, s5, s6]):
        p_s = p[sx[i,:]]
        if p_s.size > 0:
            reformat4[i, j] = max(p_s)
    reformat4[i, 6] = reformat4[i, : 6].sum()
    '''
    p_s1 = p[s1[i,:]]
    p_s2 = p[s2[i,:]]
    p_s3 = p[s3[i,:]]
    p_s4 = p[s4[i,:]]
    p_s5 = p[s5[i,:]]
    p_s6 = p[s6[i,:]] 
    # print(p_s1, p_s2, p_s3, p_s4, p_s5, p_s6)
    
    if(p_s1.size==0 or p_s2.size==0 or p_s3.size==0 or p_s4.size==0 or p_s5.size==0 or p_s6.size==0):
        continue
    reformat4[i,:]=np.array([max(p_s1), max(p_s2), max(p_s3), max(p_s4), max(p_s5), max(p_s6)])
    '''

sofa_component = ['Respiration_SOFA', 'Coagulation_SOFA', 'Liver_SOFA', 'Cardiovascular_SOFA', 'CNS_SOFA', 'Renal_SOFA']
for i, sofa in enumerate(sofa_component):
    reformat4t[sofa] = reformat4[:,i]
reformat4t['SOFA'] = reformat4[:,6]


sofa_list = reformat4t[sofa_component].values
sofa_sum = reformat4t['SOFA'].values
delta = sofa_list.sum(1) - sofa_sum
print(delta.min(), delta.max())
print(sofa_sum.min(), sofa_sum.max())





df = reformat4t

for k in ['FiO2_1', 'PEEP', 'TidalVolume']:
    values = df[k].values
    values[np.isnan(values)] = np.nanmean(values)
    print(k, values.min(), values.max())

icustayid_selected = []
for icustayid in tqdm(df['icustayid'].unique()):
    data = df[df['icustayid'] == icustayid]
    age = data['age'].values[0]
    if age < 18 * 365:
        continue
    weight = data['Weight_kg'].values
    if weight.min() < 40:
        continue

    if data['mechvent'].values.sum() > 6:
        icustayid_selected.append(icustayid)
print('There are {:d} patients'.format(len(icustayid_selected)))
print('There are {:d} lines'.format(len(df)))

# df = df.loc[df['icustayid'] in icustayid_selected]
df = df[df['icustayid'].isin(icustayid_selected)]
print(len(df['icustayid'].unique()))
print('There are {:d} lines'.format(len(df)))

weight = df['Weight_kg'].values
df['TidalVolume'] = df['TidalVolume'].values / df['Weight_kg'].values

df.to_csv('../data/mechvent_cohort.csv', index=False)

