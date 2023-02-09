import pandas as pd
import numpy as np
import os
import time
import math
import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel('6-20-22 match 75 location demo 3class 11-27-21.xlsx')
list_pid_temp = df['AWAREID'].values.tolist()
list_pid=[]
for i in range(len(list_pid_temp)):
    if(len(list_pid_temp[i])==3):
        list_pid.append(list_pid_temp[i][0])
        list_pid.append(list_pid_temp[i][1])
        list_pid.append(list_pid_temp[i][2])
    elif(len(list_pid_temp[i])==2):
        list_pid.append(list_pid_temp[i][0])
        list_pid.append(list_pid_temp[i][1])
    else:
        list_pid.append(list_pid_temp[i][0])


##Enter the data set for which latitude and longitude are required
file_name=''
df_6h = pd.read_csv(file_name)

list_time = df_6h['dateTime'].to_list()
list_timestamp=[]
for i in range(len(list_time)):
    str_time =list_time[i]
    timeArray = time.strptime(str_time, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    list_timestamp.append(timeStamp)
df_6h['time_stamp']=list_timestamp

df_6h[['avg_latitude','avg_longitude',"med_latitude",'med_longitude','std_latitude','std_longitude','WTSD_latitude','WTSD_longitude','max_latitude','max_longitude','min_latitude','min_longitude']]=df_6h.apply(lambda x:('','','','','','','','','','','',''),axis=1,result_type='expand')
df_6h[['total_distance','speed_mean_sec','moving_time','number_location']]=df_6h.apply(lambda x:('','','',''),axis=1,result_type='expand')
list_pid_co=df_6h['pid'].to_list()
list_ts_co=df_6h['time_stamp'].to_list()

df_temp = df[df['ID']==list_pid_co[0]]
device_id = df_temp['AWAREID'].to_list()
device_idsr = device_id[0].split(', ')
str_name = device_idsr[0]+'_locations.csv'
#Retrieve the corresponding dataset storing latitude and longitude information
df_deviceid = pd.read_csv(str_name)

device_id_now = device_idsr[0]


for bigi in range (3):
    l=len(list_pid_co)

    for i in range (len(list_pid_co)):
    
        df_temp = df[df['ID']==list_pid_co[i]]
        device_id_str = df_temp['AWAREID'].to_list()
        if len(device_id_str)==0:
            continue
        device_id = device_id_str[0].split(', ')
        if df_6h.loc[i,'avg_latitude']=='':
            try:
                if device_id_now==device_id[bigi]:
                    df_deviceid_temp = df_deviceid.loc[df_deviceid['timestamp']>list_ts_co[i]-900]
                    df_deviceid_temp = df_deviceid_temp[df_deviceid_temp['timestamp']<=list_ts_co[i]]
                    
                    df_6h.loc[i,'avg_latitude'] = df_deviceid_temp['double_latitude'].mean()
                    df_6h.loc[i,'avg_longitude'] = df_deviceid_temp['double_longitude'].mean()
                    df_6h.loc[i,'med_latitude'] = df_deviceid_temp['double_latitude'].median()
                    df_6h.loc[i,'med_longitude'] = df_deviceid_temp['double_longitude'].median()
                    df_6h.loc[i,'std_latitude'] = df_deviceid_temp['double_latitude'].std()
                    df_6h.loc[i,'std_longitude'] = df_deviceid_temp['double_longitude'].std()
                    df_6h.loc[i,'max_latitude'] = df_deviceid_temp['double_latitude'].max()
                    df_6h.loc[i,'max_longitude'] = df_deviceid_temp['double_longitude'].min()
                    df_6h.loc[i,'min_latitude'] = df_deviceid_temp['double_latitude'].max()
                    df_6h.loc[i,'min_longitude'] = df_deviceid_temp['double_longitude'].min()
                    
                    df_6h.loc[i,'total_distance']=df_deviceid_temp['distance'].sum()
                    
                    df_deviceid_move = df_deviceid_temp.loc[(df_deviceid_temp['stationary_datapoints']==1)]
                    df_6h.loc[i,'moving_time'] = df_deviceid_move['duration'].sum()
                    if df_6h.loc[i,'total_distance']!=0:
                        all_time=df_deviceid_temp['duration'].sum()
                        df_6h.loc[i,'speed_mean_sec' ] = df_6h.loc[i,'total_distance']/all_time
                    else:
                        df_6h.loc[i,'speed_mean_sec' ] =0
                
                    nu_lo=1
                    df_deviceid_temp=df_deviceid_temp.sort_values(by='timestamp')
                    col_day=list(df_deviceid_temp.columns)
                    df_deviceid_temp=df_deviceid_temp.reset_index()[col_day]
                    if(len(df_deviceid_temp)>1):
                        for lo_num in range (1,len(df_deviceid_temp)):
                            if int(df_deviceid_temp.loc[lo_num-1,'stationary_datapoints'])==1 & int(df_deviceid_temp.loc[lo_num,'stationary_datapoints'])==0:
                                nu_lo=nu_lo+1
                        df_6h.loc[i,'number_location' ] = nu_lo
                    else:
                        df_6h.loc[i,'number_location' ] = 1                   
                
                    df_deviceid_temp = df_deviceid_temp.loc[(df_deviceid_temp['stationary_datapoints']==0)]
                    mer = np.array([df_deviceid_temp['double_latitude'].to_list(),df_deviceid_temp['timestamp'].to_list()])
                    mer_np=np.cov(mer)
                    df_6h.loc[i,'WTSD_latitude'] = math.sqrt(abs(mer_np[0][1]))
                    
                    mer2 = np.array([df_deviceid_temp['double_longitude'].to_list(),df_deviceid_temp['timestamp'].to_list()])
                    mer2_np=np.cov(mer2)
                    df_6h.loc[i,'WTSD_longitude'] = math.sqrt(abs(mer2_np[0][1]))
                    
                
                else:
                    device_id_now=device_id[bigi]
                    
                    str_name = device_id[bigi]+'_locations.csv'
                    
                    df_deviceid = pd.read_csv(str_name)
                
                    df_deviceid_temp = df_deviceid.loc[df_deviceid['timestamp']>list_ts_co[i]-300]
                    df_deviceid_temp = df_deviceid_temp[df_deviceid_temp['timestamp']<=list_ts_co[i]]
                    
                    df_6h.loc[i,'avg_latitude'] = df_deviceid_temp['double_latitude'].mean()
                    df_6h.loc[i,'avg_longitude'] = df_deviceid_temp['double_longitude'].mean()
                    df_6h.loc[i,'med_latitude'] = df_deviceid_temp['double_latitude'].median()
                    df_6h.loc[i,'med_longitude'] = df_deviceid_temp['double_longitude'].median()
                    df_6h.loc[i,'std_latitude'] = df_deviceid_temp['double_latitude'].std()
                    df_6h.loc[i,'std_longitude'] = df_deviceid_temp['double_longitude'].std()
                    df_6h.loc[i,'max_latitude'] = df_deviceid_temp['double_latitude'].max()
                    df_6h.loc[i,'max_longitude'] = df_deviceid_temp['double_longitude'].min()
                    df_6h.loc[i,'min_latitude'] = df_deviceid_temp['double_latitude'].max()
                    df_6h.loc[i,'min_longitude'] = df_deviceid_temp['double_longitude'].min()
                    df_6h.loc[i,'total_distance']=df_deviceid_temp['distance'].sum()
                    
                    df_deviceid_move = df_deviceid_temp.loc[(df_deviceid_temp['stationary_datapoints']==1)]
                    df_6h.loc[i,'moving_time' ] = df_deviceid_move['duration'].sum()
                    if df_6h.loc[i,'total_distance']!=0:
                        all_time=df_deviceid_temp['duration'].sum()
                        df_6h.loc[i,'speed_mean_sec' ] = df_6h.loc[i,'total_distance']/all_time
                    else:
                        df_6h.loc[i,'speed_mean_sec' ] =0
                
                    nu_lo=1
                    df_deviceid_temp=df_deviceid_temp.sort_values(by='timestamp')
                    col_day=list(df_deviceid_temp.columns)
                    df_deviceid_temp=df_deviceid_temp.reset_index()[col_day]
                    if(len(df_deviceid_temp)>1):
                        for lo_num in range (1,len(df_deviceid_temp)):
                            if int(df_deviceid_temp.loc[lo_num-1,'stationary_datapoints'])==1 & int(df_deviceid_temp.loc[lo_num,'stationary_datapoints'])==0:
                                nu_lo=nu_lo+1
                        df_6h.loc[i,'number_location' ] = nu_lo
                    else:
                        df_6h.loc[i,'number_location' ] = 1
                
                    df_deviceid_temp = df_deviceid_temp.loc[(df_deviceid_temp['stationary_datapoints']==0)]
                    mer = np.array([df_deviceid_temp['double_latitude'].to_list(),df_deviceid_temp['timestamp'].to_list()])
                    mer_np=np.cov(mer)
                    df_6h.loc[i,'WTSD_latitude'] = math.sqrt(abs(mer_np[0][1]))
                    mer2 = np.array([df_deviceid_temp['double_longitude'].to_list(),df_deviceid_temp['timestamp'].to_list()])
                    mer2_np=np.cov(mer2)
                    df_6h.loc[i,'WTSD_longitude'] = math.sqrt(abs(mer2_np[0][1]))
                    

            except:
                continue
df_6h.to_csv('../result/result_%s' %file_name)