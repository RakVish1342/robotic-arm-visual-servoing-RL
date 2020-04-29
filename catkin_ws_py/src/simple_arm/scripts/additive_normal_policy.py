import scipy.stats
import numpy as np
# from visual_dynamics.policies import Policy
# from visual_dynamics.spaces import AxisAngleSpace, TranslationAxisAngleSpace


class AdditiveNormalPolicy():
    def __init__(self, pol, action_space, state_space, act_std=None, reset_std=None):
        self.pol = pol
        self.action_space = action_space
        self.state_space = state_space
        self.act_std = act_std
        self.reset_std = reset_std
        self.low = np.array([-0.8,-0.8])
        self.high = np.array([0.8,0.8])

    # @staticmethod
    def truncated_normal(self,mean, std):
        if std is None:
            return mean
        std = std * (self.high - self.low) / 2.
        tn = scipy.stats.truncnorm((self.low - mean) / std, (self.high - mean) / std, mean, std)
        # if isinstance(space, (AxisAngleSpace, TranslationAxisAngleSpace)) and \
        #         space.axis is None:
        #     raise NotImplementedError
        return tn.rvs()

    def act(self, obs):
        # return self.truncated_normal(self.pol.act(obs), self.act_std, self.action_space)
        return self.truncated_normal(self.pol.act(obs), self.act_std)

    def reset(self):
        # return self.truncated_normal(self.pol.reset(), self.reset_std, self.state_space)
        return self.truncated_normal(self.pol.reset(), self.reset_std)

    def _get_config(self):
        config = super(AdditiveNormalPolicy, self)._get_config()
        config.update({'pol': self.pol,
                       'action_space': self.action_space,
                       'state_space': self.state_space,
                       'act_std': self.act_std,
                       'reset_std': self.reset_std})
        return config
