#### Bayesian Online Multivariate Changepoint Detection Algoithm 
***Student:*** Ilaria Lauzana
***Supervisors:*** Nadia Figueroa, Jose Medina

---

This repository contains the implementation of the Bayesian Online Multivariate Changepoint Detection algorithm, proposed by Ilaria Lauzana, Nadia Figueroa and Jose Medina. 

We provide 3 implementations:
- matlab
- python
- ros node written in python (online_changepoint_detector)

You can find each implementation in its corresponding folder:
##### Structure
```
.
├── README.md
└── matlab
    ├── README.md
    │   └── code
    │   └── lightspeed
└── python
    ├── python-univariate
        ├── README.md
        │   └── bayesian_changepoint_detection
    ├── python-multivariate
└── online_changepoint_detector
    ├── CMakeLists.txt
    ├── package.xml
    └── scripts
└── data
└── results - figures
└── report-project-changepoint
    ├── README.md
    ├── main.tex
    └── references
```

#### Instructions:

####Matlab
The matlab implementation is a self-contained code, no dependencies are needed. Except for the [lightspeed](http://research.microsoft.com/en-us/um/people/minka/software/lightspeed/) toolbox, which is provided within the folder.

In order to run the changepoint detector, run ``./matlab/code/gaussdemo_multi.m``.

####Python
For the python implementation, install the following python libraries for linear algebra, machine learning methods and plotting:
```
$ sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
```
Then install seaborn:
```
$ sudo pip install seaborn
```


####Ros Node
