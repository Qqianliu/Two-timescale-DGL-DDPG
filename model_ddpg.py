# time 20220624
# by liuqian
# DDPG MADDPG PPO by pytorch

from Replay_buffer import Replay_buffer
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from tensorboardX import SummaryWriter

# device 的用处就是作为tensor 或者model 被分配到的位置。表示将构建的张量或者模型分配到相应的设备上
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,args):
        super(Actor, self).__init__()
        # num_units_1 = 128
        # num_units_2 = 128
        # print(type(state_dim),type(args.num_units_1))
        self.linear_a1 = nn.Linear(state_dim, args.num_units_1)
        self.linear_a2 = nn.Linear(args.num_units_1, args.num_units_2)
        self.linear_a = nn.Linear(args.num_units_2, action_dim)
        self.reset_parameters()
        # Activation func init
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.train()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_a1.weight.data.mul_(gain)
        self.linear_a2.weight.data.mul_(gain)
        self.linear_a.weight.data.mul_(gain_tanh)

    def forward(self, input_state):
        #print("!!!!!",input_state)
        x = self.LReLU(self.linear_a1(input_state))
        #print("x",x)
        x = self.LReLU(self.linear_a2(x))
        #@print("second layer，", x)
        policy = self.tanh(self.linear_a(x))
        # print("policy，", policy)
        return policy


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,args):
        super(Critic, self).__init__()
        # num_units_1 = 128
        # num_units_2 = 128
        self.linear_c1 = nn.Linear(state_dim + action_dim, args.num_units_1)
        self.linear_c2 = nn.Linear(args.num_units_1 , args.num_units_2)
        self.linear_c = nn.Linear(args.num_units_2, 1)
        self.reset_parameters()
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.train()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        self.linear_c1.weight.data.mul_(gain)
        self.linear_c2.weight.data.mul_(gain)
        self.linear_c.weight.data.mul_(gain)

    def forward(self, input_state, input_action):
        x = self.LReLU(self.linear_c1(torch.cat([input_state, input_action], 1)))
        x = self.LReLU(self.linear_c2(x))
        x = self.linear_c(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim,args):
        self.actor = Actor(state_dim, action_dim,args).to(device)
        self.actor_target = Actor(state_dim, action_dim,args).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr_a)

        self.critic = Critic(state_dim, action_dim, args).to(device)
        self.critic_target = Critic(state_dim, action_dim,args).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr_c)
        self.replay_buffer = Replay_buffer(args.memory_size)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    # 这里是否应该有反向传播？？
    def select_action(self, state):
        #print("the state", len(state))
        #print(type(state))
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        #print(self.actor(state).cpu().data.numpy().flatten())
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, args):
        for it in range( args.update_iteration):
            # Sample replay buffer
            o, a, o_, r = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(o).to(device)
            action = torch.FloatTensor(a).to(device)
            next_state = torch.FloatTensor(o_).to(device)
            # done = torch.FloatTensor(1-d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            # self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # nn.utils.clip_grad_norm_(self.critic.parameters(), args.max_grad_norm)
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            # self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self,args):
        torch.save(self.actor.state_dict(), args.save_dir + 'actor.pth')
        torch.save(self.critic.state_dict(), args.save_dir + 'critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self,args):
        self.actor.load_state_dict(torch.load(args.save_dir + 'actor.pth'))
        self.critic.load_state_dict(torch.load(args.save_dir + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")
