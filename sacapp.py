#SAC for PDE, diffusion
from sagemaker.session import Session
from sagemaker.experiments.run import Run, load_run
from sagemaker.utils import unique_name_from_base
#import os
#import sys
#import logging
#from IPython.display import set_matplotlib_formats
#from matplotlib import pyplot as plt
import numpy as np
from numpy import prod

import pickle
import boto3
#import math
#import random

import gymnasium
import jax
import coax
import haiku as hk
import jax.numpy as jnp
#from numpy import prod
import optax
import time
import argparse
import shutup;
shutup.please()


#will need to restart due to py-pde exeuction time growth
#so be able to save/load existing nueral networks
#thought not buffers, since that would take too long
import sagemaker.session
from sagemaker import get_execution_role


class controller:
    def __init__(self, grid, state, num_sens, stepper):
        self.grid=grid
        self.num_sens=num_sens
        self.state=state
        self.control=[]
        self.stepper=stepper

#need function to reduce full space to just sensors 
def sensor_meas(sensors,domain):
    #assume senors form a square grid, so take square root of number of sensors
    #assume sensors read from middle of domain
    n_sense=int(np.sqrt(sensors))
    meas=np.empty((n_sense,n_sense))
    dimx=domain.shape[0]
    dimy=domain.shape[1]
    startx=round((dimx-n_sense)/2)
    starty=round((dimy-n_sense)/2)
    for i in range(n_sense):
        for j in range(n_sense):
            meas[i,j]=float(domain[i+startx,j+starty])
    return meas.flatten()

name = 'sac'

# the Pendulum MDP
env = gymnasium.make('Diffusion-v0')
obs, grid, state, stepper = env.reset()
env = coax.wrappers.TrainMonitor(env, name=name, tensorboard_dir=f"./data/tensorboard/{name}")

parser = argparse.ArgumentParser()

# sagemaker-containers passes hyperparameters as arguments
parser.add_argument("--layer_size", type=int, default= 4)
parser.add_argument("--n_layer", type=int, default=2)
parser.add_argument("--buffer_size", type=int, default=15000)
parser.add_argument("--maxT", type=int, default=450000)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--load_flag", type=int, default=0)
parser.add_argument("--sagemaker_session", default=None)

args = parser.parse_args()
AWS_REGION='us-east-2'
session = sagemaker.Session(boto3.session.Session(region_name=AWS_REGION))

session.boto_region_name

# pass in the sagemaker session as an argument
role = get_execution_role(sagemaker_session=session)

#role = sagemaker.get_execution_role()

region = boto3.Session().region_name

bucket = session.default_bucket()
prefix = 'sac_pde'
s3 = boto3.resource('s3')

import logging
logger = logging.getLogger('requests_throttler')
logger.addHandler(logging.NullHandler())
logger.propagate = False

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler(sys.stdout))

#hyperparmeters
layer_size=6#args.layer_size
n_layer=2#args.n_layer
buffer_size=5000#args.buffer_size
alpha=0.2#args.alpha
batch=128#args.batch
maxT=100000#args.maxT

#identifies if first run or not and if to continue training existing NNs
load_flag=0#args.load_flag

def func_pi(S, is_training):
    # a='hk.Linear(layer_size), jax.nn.relu, '
    # n=0
    # input='hk.Sequential(( '
    # while n<n_layer:
    #     input+=a
    #     n+=1
    # input+='hk.Linear(prod(env.action_space.shape) * 2, w_init=jnp.zeros), hk.Reshape((*env.action_space.shape, 2)), ))'
    # seq=eval(input)
    seq=hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(prod(env.action_space.shape) * 2, w_init=jnp.zeros),
        hk.Reshape((*env.action_space.shape,2)),
    ))
    x = seq(S)
    mu, logvar = x[..., 0], x[..., 1]
    return {'mu': mu, 'logvar': logvar}


def func_q(S, A, is_training):
    # a='hk.Linear(layer_size), jax.nn.relu, '
    # n=0
    # input='hk.Sequential(( '
    # while n<n_layer:
    #     input+=a
    #     n+=1
    # input+='hk.Linear(1, w_init=jnp.zeros), jnp.ravel ))'
    # seq=eval(input)
    seq=hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(1, w_init=jnp.zeros), jnp.ravel
    ))
    X = jnp.concatenate((S, A), axis=-1)
    return seq(X)
    

tracker=[0.,0.]

#insert controller parameters
# number of controls, control locations.  Assume control is on boundary
#here, with two control locations and 4 controls, that means that two boundaries have controls.
#In gymanisum this will be hard coded as bottom and right side, or (x,ymin) and (xmax, y)
num_sens=3*3
# num_control = 4
# control_size = 3 #number of grid points comprising controller
# control_points = np.array([8,20]) #grid x/y locations of controllers
dx=1.
dy=1.
xmax=grid.shape[0]*dx
ymax=grid.shape[1]*dy 


#insert controller class initialization
actor=controller(grid,state, num_sens, stepper)

def train(time0, total_reward, pi, q1, q2, q1_targ, q2_targ, tracer, buffer, policy_regularizer, qlearning1, qlearning2, soft_pg):
    obs, grid, state, stepper = env.reset()
    actor=controller(grid, state, num_sens, stepper)
    #insert sensor measurement and conversion here
    s=sensor_meas(num_sens,state.data)
    start=time.time()
    for t in range(250):
       #insert sensor measurement and conversion here
        s=sensor_meas(num_sens,state.data)
        #add x,y coordinates to make nn input vector s.  This means that must add x,y for all control points to s
        #i.e s dimension = #observation point measurements + 2*number of control points
        #x and y locations are control loc variables, so s will need to append each element of control loc into s
        i=0
        # for j in grid.axes_coords[1]:
            # #hard code going through x and y coor for each boundary location
            # s_full=jnp.append(s,j)
            # s_full=jnp.append(s_full, grid.axes_coords[0][1])
            # a=pi(s_full)
            # if a>5:  a=np.array([5.0])
            # elif a<-5:  a=-np.array([5.0])
            # state.data[i,1]=a.item()
            # i+=1
#apply control one row in from boundary to improve computation speed by allowing periodic bcs
        #not very physical, but this is a proof of concept
        #actor.state.data=state.data[:,1]
        a=pi(s)
        
        actor=a#state.data[:,1]
        
        s_next, r, done, truncated, info = env.step(actor)
        total_reward=r
        tracer.add(s, a, r, done)
        if done:  break
        # trace rewards and add transition to replay buffer
        # trace rewards and add transition to replay buffer
        
        while tracer:
            buffer.add(tracer.pop())

        # learn
        if len(buffer) >= 5000:
            transition_batch = buffer.sample(batch_size=128)

            # init metrics dict
            metrics = {}

            # flip a coin to decide which of the q-functions to update
            qlearning = qlearning1 if jax.random.bernoulli(q1.rng) else qlearning2
            metrics.update(qlearning.update(transition_batch))

            # delayed policy updates
            if env.T >= 7500 and env.T % 4 == 0:
                metrics.update(soft_pg.update(transition_batch))

            env.record_metrics(metrics)

            # sync target networks
            q1_targ.soft_update(q1, tau=0.001)
            q2_targ.soft_update(q2, tau=0.001)
        total_reward=r
        if done or truncated:
            break
        # time_check=time.time()-start
        # if time_check>100:
        #     #when py-pde calculation time diverges, save NNs and terminate instance to start over again fresh.
        #     key='piNN.pkl'
        #     piNN = pickle.dumps(pi) 
        #     s3.Object(bucket,key).put(Body=piNN)
        #     key='q1NN.pkl'
        #     q1NN = pickle.dumps(pi) 
        #     s3.Object(bucket,key).put(Body=q1NN)
        #     key='q2NN.pkl'
        #     q2NN = pickle.dumps(pi) 
        #     s3.Object(bucket,key).put(Body=q2NN)
        #     return time_check
        state.data = s_next.data
    tracker.append([time.time(), r])


if __name__ == "__main__":
    
    time0=time.process_time()
    total_reward=0.
    
    # main function approximators
    # if load_flag==1:
    #     pi=pickle.loads(s3.Bucket(bucket).Object("piNN.pkl").get()['Body'].read())
    #     q1=pickle.loads(s3.Bucket(bucket).Object("q1NN.pkl").get()['Body'].read())
    #     q2=pickle.loads(s3.Bucket(bucket).Object("q2NN.pkl").get()['Body'].read())
    # else:
    pi = coax.Policy(func_pi, env)
    q1 = coax.Q(func_q, env, action_preprocessor=pi.proba_dist.preprocess_variate)
    q2 = coax.Q(func_q, env, action_preprocessor=pi.proba_dist.preprocess_variate)

    # target network
    q1_targ = q1.copy()
    q2_targ = q2.copy()

    # experience tracer
    tracer = coax.reward_tracing.NStep(n=5, gamma=0.9, record_extra_info=True)
    buffer = coax.experience_replay.SimpleReplayBuffer(capacity=buffer_size)

    policy_regularizer = coax.regularizers.NStepEntropyRegularizer(pi,
                                                                   beta=alpha / tracer.n,
                                                                   gamma=tracer.gamma,
                                                                   n=[tracer.n])

    # updaters (use current pi to update the q-functions and use sampled action in contrast to TD3)
    qlearning1 = coax.td_learning.SoftClippedDoubleQLearning(
        q1, pi_targ_list=[pi], q_targ_list=[q1_targ, q2_targ],
        loss_function=coax.value_losses.mse, optimizer=optax.adam(1e-3),
        policy_regularizer=policy_regularizer)
    qlearning2 = coax.td_learning.SoftClippedDoubleQLearning(
        q2, pi_targ_list=[pi], q_targ_list=[q1_targ, q2_targ],
        loss_function=coax.value_losses.mse, optimizer=optax.adam(1e-3),
        policy_regularizer=policy_regularizer)
    soft_pg = coax.policy_objectives.SoftPG(pi, [q1_targ, q2_targ], optimizer=optax.adam(
        1e-3), regularizer=coax.regularizers.NStepEntropyRegularizer(pi,
                                                                     beta=alpha / tracer.n,
                                                                     gamma=tracer.gamma,
                                                                     n=jnp.arange(tracer.n)))
                                                                     
    while env.T < maxT:
        timer=train(time0, total_reward, pi, q1, q2, q1_targ, q2_targ, tracer, buffer, policy_regularizer, qlearning1, qlearning2, soft_pg)
        print(timer)
        
        #t+=1