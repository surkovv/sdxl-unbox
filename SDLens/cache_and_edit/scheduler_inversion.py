from typing import List, Optional, Tuple, Union
import torch
from diffusers.configuration_utils import register_to_config
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerOutput


class FlowMatchEulerDiscreteSchedulerForInversion(FlowMatchEulerDiscreteScheduler):

    @register_to_config
    def __init__(self, inverse: bool, **kwargs):
        super().__init__(**kwargs)
        self.inverse = inverse


    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        if self.inverse:
            next_sample = sample + (sigma - sigma_next) * model_output
            # Cast sample back to model compatible dtype
            next_sample = next_sample.to(model_output.dtype)
            # upon completion increase step index by one
            self._step_index -= 1

            if not return_dict:
                return (next_sample,)

            return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=next_sample)
        else:
            prev_sample = sample + (sigma_next - sigma) * model_output
            # Cast sample back to model compatible dtype
            prev_sample = prev_sample.to(model_output.dtype)
            # upon completion increase step index by one
            self._step_index += 1

            if not return_dict:
                return (prev_sample,)

            return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
