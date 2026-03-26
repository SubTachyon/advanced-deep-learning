## Written Summary

I loaded the spectra and labels in Python and used PyTorch to build a simple CNN that predicts `t_eff`, `log_g`, and `fe_h`. Before training, I log-normalized the spectra and standardized both the input data and the labels. I also plotted a few spectra to get a better idea of what the data looked like.

At first, the results were not that good, so I tried changing the number of epochs and also changing the `AdaptiveAvgPool1d` layer. That improved the model a lot. My better result was about `56.5` for `t_eff`, `0.1115` for `log_g`, and `0.0458` for `fe_h` (in terms of MAE).

The main challenge was finding model settings that worked better, because the first version was too weak. If I was to spend more time on this, I would try a few more architectures and tune the training settings more.
