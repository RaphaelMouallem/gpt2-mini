class TrainerConfig:
    def __init__(self, **kwargs):
        self.max_steps = kwargs.get('max_steps', 1000)
        self.step_accum = kwargs.get('step_accum', 4)
        self.log_interval = kwargs.get('log_interval', 10)
        self.eval_interval = kwargs.get('eval_interval', 100)
        self.eval_steps = kwargs.get('eval_steps', 10)
        self.checkpoint_interval = kwargs.get('checkpoint_interval', 500)
        self.checkpoint_dir = kwargs.get('checkpoint_dir', './checkpoints')
        self.device = kwargs.get('device', 'cpu')