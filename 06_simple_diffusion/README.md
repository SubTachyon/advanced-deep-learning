# Written Summary

For this exercise I made a very small 1D diffusion model. The data was just one number sampled from two Gaussians centered around -4 and 4. The network was a simple MLP that took the current noisy value and the timestep, and learned to predict the noise that had been added.

After training for 50 epochs, the generated samples looked quite close to the real distribution. The model learned the two peaks and also roughly matched that the right peak should be larger. The loss went down quickly and then mostly flattened out, so training longer did not seem very useful.

The main challenge was getting the reverse diffusion sampling to behave properly and making the plots easy to understand. The result is good enough for this simple case, but it could probably be improved with more tuning, a better noise schedule, or a slightly better model.
