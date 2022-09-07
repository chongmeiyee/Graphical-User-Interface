#!/usr/bin/env python
# coding: utf-8

# In[3]:


from tkinter import *
import numpy as np
import scipy.stats as ss
import sys

tk=Tk()
tk.title('Lookback Fixed Call Option Pricing')
tk.geometry('500x500')

def LSMC():
     S0 = float(S0_entry.get())  
     K = float(K_entry.get())  
     T = float(T_entry.get())  
     r = float(r_entry.get())   
     Sigma = float(Sigma_entry.get()) 
     N = int(N_entry.get())
     paths = int(P_entry.get())
    
     order = 2
     dt = T/(N-1)          #切割時間
     df = np.exp(-r * dt)  #折現因子
     
     #產生股價的過程(lognormal dist variable)
     #stock price follow geometric brownian motion   
     X0 = np.zeros((paths,1))
     increments = ss.norm.rvs(loc=(r - Sigma**2/2)*dt, scale=np.sqrt(dt)*Sigma, size=(paths,N-1))
     X = np.concatenate((X0,increments), axis=1).cumsum(1) 
     S = S0 * np.exp(X) 

     P = np.zeros_like(S) #產生空array存取payoff
     #計算payoff
     if (var1.get() == 1):  
       for i in range(0,paths):
        for j in range(1,N-1):
          if (S[i,1] >= K):
            P[i,1]=S[i,1]-K
          Smax=max(S[i,:j+1])  
          if (S[i,j+1] >= Smax):
            Smax=S[i,j+1]  
            P[i,j+1]=np.maximum(Smax-K,0)
          else: P[i,j+1]=np.maximum(Smax-K,0)
     OptV = np.zeros_like(P)            # 產生和payoff同dimension的array存取option value
     OptV[:,-1] = P[:,-1]
    # Least Square Monte Carlo
     for t in range(N-2, 0, -1):
        
        paths1 = P[:,t] > 0    #在t時In-The-Money的payoff
        reg = np.polyfit( S[paths1, t], OptV[paths1, t+1] * df, 2) 
        # polynomial regression：將對應的t+1時選擇權價值（Y）和t時ITM的payoff（X）作迴歸估計
        HoldV = np.polyval( reg, S[paths1,t] )                             
        # 如果Holding Value<exercise value，則exercise
        exercise = np.zeros( len(paths1), dtype=bool)
        exercise[paths1] = P[paths1,t] > HoldV
        OptV[exercise,t] = P[exercise,t] #t時執行，存取payoff
        OptV[exercise,t+1:] = 0 #t時執行，t+1時選擇權價值為0
        discount_path = (OptV[:,t] == 0) 
        OptV[discount_path,t] = OptV[discount_path,t+1] * df
     result = np.mean(OptV[:,1]) * df  # 
     result_label['text']=result

def clear_text():
    S0_entry.delete(0,END)
    K_entry.delete(0,END)
    r_entry.delete(0,END)
    Sigma_entry.delete(0,END)
    T_entry.delete(0,END)
    N_entry.delete(0,END)
    P_entry.delete(0,END)
    var1.set(0)
    


Label(tk,text="Underlying Price").grid(row=0,sticky=W)
S0_entry=Entry(tk)
S0_entry.grid(row=0,column=1)

Label(tk,text="Strike Price").grid(row=1,sticky=W)
K_entry=Entry(tk)
K_entry.grid(row=1,column=1)

Label(tk,text="Interest Rate").grid(row=2,sticky=W)
r_entry=Entry(tk)
r_entry.grid(row=2,column=1)

Label(tk,text="Sigma").grid(row=3,sticky=W)
Sigma_entry=Entry(tk)
Sigma_entry.grid(row=3,column=1)

Label(tk,text="Time to Maturity").grid(row=4,sticky=W)
T_entry=Entry(tk)
T_entry.grid(row=4,column=1)

Label(tk,text="Steps").grid(row=5,sticky=W)
N_entry=Entry(tk)
N_entry.grid(row=5,column=1)

Label(tk,text="Paths").grid(row=6,sticky=W)
P_entry=Entry(tk)
P_entry.grid(row=6,column=1)

var1=IntVar(tk)
c1=Checkbutton(tk,text="Call",variable=var1, onvalue=1, offvalue=0)
c1.grid(row=7,column=1,sticky=W+E,pady=2)

Label(tk,text="").grid(row=8,sticky=W)

button1=Button(tk,text="Option Price",command=LSMC)
button1.grid(row=10,column=1)
result_label=Label(tk,text="price")
result_label.grid(row=10,column=2)

Label(tk,text="").grid(row=11,sticky=W)
button2=Button(tk,text="      Clear     ",command=clear_text)
button2.grid(row=11,column=1)



tk.mainloop()


# In[ ]:





# In[ ]:




