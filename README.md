# MADDPG-Congestion-Control-Based-on-OMNETPP

## Videos
Now, the experiment video can be accessed by downloading them from ExperimentsVideo.zip

Since we use LFS to upload videos, if you want to clone project including videos, use ```git lfs clone``` to pull this repo instead of ```git clone```.

If you only want to access the code, use ```git clone```, and it won't pull videos from LFS. 

## Code
Source code of algorithm can be accessed.  

#### I'll refactor the whole project soon.


## Run and Train Your Model

By the compiled MADDPG.exe, you can train your own model !!!

```import this repo to omnet++, choose MADDPG.exe, then [run as] --> [run configuration].```

```Choose old.ini as the ini file, you can config by revising old.ini file. Don't forget to choose cmdEnv when training model```

Sadly, it's unconvienient that you should also run one algorithm with python after you launched a omnet++ simulation in it's ide.

Due to some error of omnetpp, I can not run this MADDPG.exe in python with multi-thread support.

All the communication between these two processes is implemented by .json file. 

Actually, the time consuming is mainly from the simulation, and I don't optimize it yet.

The requirement is ```gym == 0.10.5, tensorflow == 1.8.0, matplotlib and python == 3.6```


## Other Info

The MADDPG algorithm is revised from OpenAI https://github.com/openai/maddpg. Visit their website for more details.

Any question can contact me at: quanyili0057@gmail.com








