import experiment_manager as xpm
from pathlib import Path

from os.path import join

#construct absolute path to result directory

result_path = join(Path(os.getcwd()).resolve().parents[1], 'results/noniid')

list_of_experiments = xpm.get_list_of_experiments(result_path)
print(list_of_experiments[0].results)