import pickle
import matplotlib.pyplot as plt
import os 
output_dir = 'out'
exps_name = os.listdir(output_dir)
fig, ax = plt.subplots(figsize=(10, 6))
for exp_name in exps_name:
    ## skip if not a directory
    if not os.path.isdir(os.path.join(output_dir, exp_name)):
        continue
    file_names = os.listdir(os.path.join(output_dir, exp_name))
    for file_name in file_names:
        if file_name.endswith('.pkl'):
            with open(os.path.join(output_dir, exp_name, file_name), 'rb') as f:
                data = pickle.load(f)
            print(len(data['steps']))
            if len(data['steps']) != 11:
                continue
            label = exp_name.split('_')[-1]
            ax.plot(data['steps'], data['val_losses'], label=label)
plt.legend()
plt.yscale('log')
plt.xlabel('iter', fontsize=14)
plt.ylabel('val loss', fontsize=14)
plt.title('val loss', fontsize=14)
plt.legend(fontsize=14,title = "Number of Synchronous Operations")
plt.savefig(os.path.join(output_dir, 'val_loss.png'))
plt.close()