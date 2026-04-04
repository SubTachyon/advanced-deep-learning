## Written Summary

For this exercise I took my CNN from the previous lesson and modified it so that it predicts not only the three target values (`t_eff`, `log_g`, and `fe_h`), but also an uncertainty for each of them.

In the previous version, the network only had 3 output nodes and I trained it with MSE loss. Here I changed the last layer so it outputs 6 values instead: 3 for the predicted means and 3 for the predicted uncertainties, written as `log_sigma`. I kept most of the rest of the pipeline the same as before, so I still used the log-transformed spectra, split the data into training/validation/test sets, standardized everything using the training set only, and trained a 1D CNN on the GALAH spectra. The main difference was replacing the old loss with the Gaussian negative log-likelihood loss, because that makes it possible for the network to learn both the prediction and its uncertainty at the same time.

To check how well it worked, I looked at the normal predicted-vs-true plots, but also at the pull distributions. From the lesson, the idea is that if the uncertainty prediction is good, the pull distribution should look roughly like a standard normal distribution.

The results looked pretty decent overall. The predicted-vs-true plots for all three labels followed the diagonal reasonably well, so the model is definitely learning the parameters and not just guessing randomly. The uncertainty part also seemed to work reasonably well overall. The pull distributions were fairly close to a standard normal distribution, which suggests that the predicted uncertainties were at least in the right range.

One thing I found a bit confusing at first was the whole idea of a neural network predicting how uncertain it is, because it feels slightly strange that the model is giving both an answer and an error estimate. Another annoying part was making sure the uncertainty output stayed sensible, which is why I used `log_sigma` instead of predicting sigma directly, and also clamped it to a fixed range. I also had to be careful when undoing the normalization at the end, because the sigmas need to be rescaled too, not just the predicted means.

So overall I think the exercise worked. I managed to build on my previous CNN and extend it into a version that predicts both the stellar parameters and their uncertainties.
