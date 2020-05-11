class Linear_decay:
    def __init__(self, lr_init, lr_term, ep_max):
        """"Returns an object calculating linear decay factors that's callable with one parameter (episode). After 
        ep_max is reached, lr plateaus at lr_term. It is a callable so it can be pickled but still statically 
        parameterized.
            params:
                lr_init: initial learning rate (episode 1)
                lr_term: final learning rate
                ep_max: when to reach final lr
        """""
        self.lr_init = lr_init
        self.lr_term = lr_term
        self.ep_max = ep_max
    def __call__(self, ep):
        return 1-ep* (1-(self.lr_term/self.lr_init)) /self.ep_max if ep <self.ep_max else self.lr_term/self.lr_init
