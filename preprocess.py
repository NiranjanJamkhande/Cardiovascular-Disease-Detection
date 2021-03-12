# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:20:39 2021

@author: Admin
"""

import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\Admin\Desktop\Cardiovascular Disease Detection\df.csv', sep = ';')
df.isnull().sum()
df = df.drop(columns = ['id'])

df.info()

import matplotlib.pyplot as plt
import seaborn as sns
###### age column
df['age'] = df['age']/ 365
sns.boxplot(df['age'])
df['age'].describe()

##### gender column
df['gender'].value_counts()
sns.countplot(df['gender'])
df['gender'].describe()
df['gender'] = df['gender'].astype('category')


##### height column
df['height'] = df['height']/30
sns.boxplot(df['height'])
df['height'].describe()
a = df[df['height'] > 6.5]
df['height'] = np.where(df['height'] > 6.5, 6.5, df['height'])
b = df[df['height'] < 4.5]
df['height'] = np.where(df['height'] < 4.5, 4.5, df['height'])


##### weight column
sns.boxplot(df['weight'])
df['weight'].describe()
c = df[df['weight'] < 50]
df['weight'] = np.where(df['weight'] < 50, 50, df['weight'])
d = df[df['weight'] > 120]
df['weight'] = np.where(df['weight'] > 120, 120, df['weight'])


##### ap_hi column
sns.boxplot(df['ap_hi'])
df['ap_hi'].describe()
e = df[df['ap_hi'] > 150]
df['ap_hi'] = np.where(df['ap_hi'] > 150, 150, df['ap_hi'])
f = df[df['ap_hi'] < 100]
df['ap_hi'] = np.where(df['ap_hi'] < 100, 100, df['ap_hi'])


#####  ap_lo column
sns.boxplot(df['ap_lo'])
df['ap_lo'].describe()
g = df[df['ap_lo'] < 60]
df['ap_lo'] = np.where(df['ap_lo'] < 60, 60, df['ap_lo'])
h = df[df['ap_lo'] > 100]
df['ap_lo'] = np.where(df['ap_lo'] > 100, 100, df['ap_lo'])



#####  cholesterol column
df['cholesterol'].value_counts()
sns.countplot(df['cholesterol'])
df['cholesterol'].describe()


##### gluc column
df['gluc'].value_counts()
sns.countplot(df['gluc'])
df['gluc'].describe()


#####  smoke column
df['smoke'].value_counts()
sns.countplot(df['smoke'])
df['smoke'].describe()


#### alco column
df['alco'].value_counts()
sns.countplot(df['alco'])
df['alco'].describe()


#### active column
df['active'].value_counts()
sns.countplot(df['active'])
df['active'].describe()


##### cardio column
df['cardio'].value_counts()
sns.countplot(df['cardio'])
df['cardio'].describe()

##################################################
df.info()



