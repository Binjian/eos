#!/usr/bin/env python
# coding: utf-8

# In[1]:


import threading
import time


# In[2]:

signal_lock = threading.Lock()
program_exit = False
def step1(evt_main, evt_step):
    global signal_lock, program_exit
    th_terminate = False
    print("step 1: enter thread")
    while not th_terminate:
        print("step 1: enter while")
        with signal_lock:
            print("step 1: enter lock")
            if program_exit:
                th_terminate = True
                continue
        print("step 1: wait")
        evt_main.wait()
        print("step 1: start")
        evt_main.clear()
        print("step 1: clear main and sleep")
        time.sleep(2)
        evt_step.set()
        print("step 1: wake and set step")

    print("step 1: terminate")

# In[3]:


def step2(evt_step):
    global signal_lock, program_exit
    th_terminate = False
    print("step 2: enter thread")
    while not th_terminate:
        print("step 2: enter while")
        with signal_lock:
            print("step 2: enter lock")
            if program_exit:
                th_terminate = True
                continue
        print('Step 2: wait')
        evt_step.wait()
        print('Step 2: wake and clear step')
        evt_step.clear()

    print("step 2: terminate")


# In[6]:


evt_step = threading.Event()
evt_main = threading.Event()
thr_step1 = threading.Thread(target=step1,args=[evt_main, evt_step])
thr_step2 = threading.Thread(target=step2,args=[evt_step])
thr_step1.start()
thr_step2.start()

i = 0
while i < 3:
    evt_main.set()
    print('Main: set main and sleep')
    time.sleep(5)
    i = i+1
    print(f'Main: wake. i={i}')

print('Main: Set flag program_exit')
with signal_lock:
    program_exit = True

print('Main: set main evt')
evt_main.set()
print('Main: thread 1 join')
thr_step1.join()
print('Main: thread 2 join')
thr_step2.join()
print('Main: terminate')
