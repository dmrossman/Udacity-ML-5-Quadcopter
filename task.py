import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 1

        # self.state_size = self.action_repeat * 6
        self.state_size = self.action_repeat * 3
        self.action_low = 0
        self.action_high = 900    # test
        self.action_size = 1      # what if I make all the rotor speeds the same?

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # Why is the reward 1.0 - 0.3 * |actual - target|?
        # I get the abs(actual - target), but why multiply it by 0.3 
        # and subtract it from one?
        # So for a perfect reward, you get 1
        # For distances from 0 to 3.3, you get a reward that drops from 1 to 0.01
        # For distances greater that 3.3, the reward becomes negative and slowly gets
        # more negative.  It is just a linear reward with a slope of -0.3 and an 
        # intercept of 1.0
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        # What if the reward was positive if I am heading in the right direction
        # Make it smaller as we get close to the target
        distance = (((self.sim.pose[:3] - self.target_pos)**2).sum())**0.5
        # reward = ((self.target_pos - self.sim.pose[:3]) * self.sim.v).sum()
        
        # I read that ideally the reward should be between -1 and 1.  Let's 
        # try normalized, squared distance
        reward = 1 - distance / 10.0
        

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            #####  
            #####  Testing if model learns faster with only one rotor speed
            #####
            rotor_speeds = rotor_speeds * 4
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        # For the starting state, take the first position and replicate
        # it the action_repeat times (3 as the default)
        # What if I make the state smaller?
        # state = np.concatenate([self.sim.pose] * self.action_repeat) 
        state = np.concatenate([self.sim.pose[:3]] * self.action_repeat) 
        return state