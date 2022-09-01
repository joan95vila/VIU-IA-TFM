import pandas as pd
import os
import matplotlib.pyplot as plt

print('Stats has started')

episode_start = 0
episode_end = None

stats = pd.read_csv('CSVLoggerV2.log', sep=',')

loss_kl = stats['_calculate_kl_loss'][episode_start:episode_end]
loss_rec = stats['_calculate_reconstruction_loss'][episode_start:episode_end]
loss = stats['loss'][episode_start:episode_end]

loss_kl_val = stats['val__calculate_kl_loss'][episode_start:episode_end]
loss_rec_val = stats['val__calculate_reconstruction_loss'][episode_start:episode_end]
loss_val = stats['val_loss'][episode_start:episode_end]

fig, ax = plt.subplots(nrows=3, num='Current training')

fig.suptitle('Learning curve', fontsize=16)

ax[0].set_title('Loss')
ax[0].plot(loss, color='b', label='Training loss')
ax[0].plot(loss_val, color='r', label='Validation loss')

ax[1].set_title('Rec loss')
ax[1].plot(loss_rec[:], color='b')
ax[1].plot(loss_rec_val[:], color='r')

ax[2].set_title('KL loss')
ax[2].plot(loss_kl, color='b')
ax[2].plot(loss_kl_val, color='r')

fig.legend()

plt.show()
