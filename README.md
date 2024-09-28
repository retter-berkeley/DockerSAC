Files used to copmlie docker container for Soft Actor Critic applied to Ordinary Differential Equation based gymansiums.
This does not contain the files necessary for the gymansium.  Those are in a seaperate project, and users will need to include the gymansium within the container on their own
My approach was to add the ODE gymanisum files into gymansium's classical control package, and include dynamics scripts (called by solve_ivp) in another directory within the container

Conents:
Docker file
requirements
sacapp.py (train.py file)
Notebook used in Sagemaker to compile and run custom container and hyperparameter tuning job
