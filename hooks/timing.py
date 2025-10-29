class TimedHook:
    def __init__(self, hook_fn, total_steps, apply_at_steps=None):
        self.hook_fn = hook_fn
        self.total_steps = total_steps
        self.apply_at_steps = apply_at_steps
        self.current_step = 0

    def identity(self, module, input, output):
        return output

    def __call__(self, module, input, output):
        if self.apply_at_steps is not None:
            if self.current_step in self.apply_at_steps:
                self.__increment()
                return self.hook_fn(module, input, output)
            self.__increment()
            return self.identity(module, input, output)

        return self.identity(module, input, output)

    def __increment(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        else:
            self.current_step = 0
