
import pickle5 as pickle
import sys
import os

sys.path.append('../src')
from utils import utils, masking_generator
from utils.slack import notification_slack


visual_bbox = utils.init_visual_bbox()

file_names = os.listdir("./data/preprocessing_shared/10000_split/")
for file_name in file_names[7:10]:
    with open(f"./data/preprocessing_shared/10000_split/{file_name}", 'rb') as f:
        data = pickle.load(f)
    print(len(data), flush=True)
    notification_slack(f"{file_name}: start add alignment labels")
    try:
        for d in data:
            al_labels = utils.create_alignment_label(
              visual_bbox,
              text_bbox=d["bbox"], 
              bool_mi_pos=d["bool_masked_pos"][0],
              )
            d["alignment_labels"] = al_labels
    except Exception as e:
        print(e)
    notification_slack(f"{file_name}: added alignment labels and saving....")
    try:
        with open(f"./data/preprocessing_shared/wpa_10000/{file_name}", 'wb') as f:
            pickle.dump(data, f, protocol=5)
    except Exception as e:
        print(e, flush=True)
        notification_slack(f"{file_name}:{e}.")
        with open(f"./data/preprocessing_shared/error_file.pkl", 'wb') as f:
            pickle.dump(data, f, protocol=5)

    notification_slack(f"{file_name}: saved!!!!!.")
    print("saved", flush=True)
    data.clear()

# file_name  = file_names[2]
# with open(f"./data/preprocessing_shared/10000_split/{file_name}", 'rb') as f:
#     data = pickle.load(f)
# print(len(data), flush=True)
# notification_slack(f"{file_name}: start add alignment labels")
# try:
#     for d in data:
#         al_labels = utils.create_alignment_label(
#             visual_bbox,
#             text_bbox=d["bbox"], 
#             bool_mi_pos=d["bool_masked_pos"][0],
#             )
#         d["alignment_labels"] = al_labels
# except Exception as e:
#     print(e)
# notification_slack(f"{file_name}: added alignment labels and saving....")
# try:
#     with open(f"./data/preprocessing_shared/wpa_10000/{file_name}", 'wb') as f:
#         pickle.dump(data, f, protocol=5)
# except Exception as e:
#     print(e, flush=True)
#     notification_slack(f"{file_name}:{e}.")
    

notification_slack(f"{file_name}: saved!!!!!.")
print("saved", flush=True)
data.clear()