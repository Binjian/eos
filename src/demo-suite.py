#!/usr/bin/env python
# starts programs until you type 'q'
import os
import time

while True:
    pid = os.fork()
    if pid == 0:  # copy process
        time.sleep(1)
        os.execlp("python", "python", "draw_lane.py")  #  draw lane on the road
        assert False, "error starting program"  # shouldn't return
    else:
        print("Child is", pid)
        os.execlp("python", "python", "demo.py")  #  draw lane on the road
        if input() == "q":
            break
