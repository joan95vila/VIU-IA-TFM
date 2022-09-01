import pandas as pd
import os
import matplotlib.pyplot as plt

print('Stats has started')

episode_start = 0
episode_end = None

stats = pd.read_csv('CSVLogger.log', sep=',')

loss = stats['loss'][episode_start:episode_end]
val_loss = stats['val_loss'][episode_start:episode_end]
# print(loss)

fig = plt.figure(num='Current training')

plt.title('Learning curve')

plt.xlabel('Epoch')
plt.ylabel('Loss value')

plt.plot(loss, color='b', label='Training loss')
plt.plot(val_loss, color='r', label='Validation loss')

fig.legend()

plt.show()
