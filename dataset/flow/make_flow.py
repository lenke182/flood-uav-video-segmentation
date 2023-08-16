# Prepares and splits the dataset into training, validation and test dataset

import os
import numpy as np
from skimage import io
import pandas

colors = np.loadtxt("list/colors.txt").astype('uint8')
num_classes = len(colors)

video_segment_start_frame = {
    "florida-01": 13037,
    "florida-02": 2389,
    "florida-03": 6137,
    "florida-04": 23626,
    "florida-05": 27884,
    "florida-06": 30737,
    "florida-07": 8746,
    "florida-08": 15048,
    "florida-09": 21209,
    "texas-01": 0,
    "florida-u": 0
}

video_speed = {
    "florida-01": 1.0,
    "florida-02": 1.0,
    "florida-03": 1.0,
    "florida-04": 3.0,
    "florida-05": [
        {"start": 0, "speed": 3.0},
        {"start": 515, "speed": 1.5},
        {"start": 1060, "speed": 2.0}
    ],
    "florida-06": 1.0,
    "florida-07": 1.5,
    "florida-08": 1.5,
    "florida-09": 1.0,
    "texas-01": 1.0,
    "florida-u": 1.0
}

video_unsupervised_index = {
    "florida-01": [],
    "florida-02": [3,4,5,6,8,9,10,11,13,14,15,16,19,20,23,24,26,27,28,29,33,34,35,37,38,47,48,49,50,52,53,54,55,56,58,59,63,65,68,69,70,71,74,77,81,86,87,88,91,96,98,102],
    "florida-03": [],
    "florida-04": [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
    "florida-05": [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50],
    "florida-06": [2,4,6,8,10,12,14,16,18,20,22,24,26,28],
    "florida-07": [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,95],
    "florida-08": [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
    "florida-09": [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54],
    "texas-01": [],
    "florida-u": [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,335,336,337,338,339,340,341,342,343,344,345,346,347,348,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,574,575,576,577,578,579,580,581,582,583,584,585,586,587,588,589,590,591,592,593,594,595,651,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,669,670,671,672,673,674,675,676,677,678,679,680,681,682,683,684,685,686,687,688,689,690,691,692,693,694,695,696,697,698,699,700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723,724,725,726,727,728,729,730,731,732,733,734,735,736,737,738,739,740,741,742,743,744,745,746,747,748,749,750,751,754,755,756,757,758,759,760,761,762,763,764,765,766,767,768,769,770,771,772,773,774,775,776,777,778,779,780,781,782,783,784,785,786,787,788,789,790,791,792,793,794,795,796,797,798,799,800,801,802,803,804,805,806,807,808,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,832,833,834,835,836,837,838,839,840,841,905,906,907,908,909,910,911,912,913,914,915,916,917,918,919,920,921,922,923,924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,1045,1046,1047,1048,1049,1050,1051,1052,1053,1054,1057,1058,1059,1060,1061,1062,1063,1064,1065,1066,1067,1068,1069,1070,1071,1072,1073,1074,1075,1076,1077,1078,1079,1080,1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095,1096,1097,1098,1099,1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,1129,1131,1133,1135,1137,1139,1141,1143,1145,1147,1149,1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,1161,1162,1163,1164,1165,1166,1167,1168,1169,1170,1171,1172,1173,1174,1175,1176,1177,1178,1179,1180,1181,1182,1183,1184,1185,1186,1187,1188,1189,1190,1224,1225,1226,1262,1263,1264,1265,1266,1267,1268,1269,1270,1271,1272,1273,1274,1275,1276,1277,1278,1279,1280,1281,1282,1283,1284,1285,1286,1287,1288,1289,1290,1291,1292,1293,1294,1295,1296,1297,1298,1299,1300,1301,1302]
}

def get_global_frame_id(video, i):
    # Identify the frame id from the filename
    relative_frame_id = (i-1)*25
    if type(video_speed[video]) == list:
        chapter = None
        for k in range(len(video_speed[video])):
            if (k < len(video_speed[video])-1 and relative_frame_id >= video_speed[video][k]["start"] and relative_frame_id < video_speed[video][k+1]["start"]) or \
                (k == len(video_speed[video])-1 and relative_frame_id >= video_speed[video][k]["start"]):
                chapter = k
                break

        if chapter is None:
            raise Exception("Didn't find chapter for i=" + str(i))

        frame_id = 0
        for p in range(chapter+1):
            if p == chapter:
                frame_id += int(video_speed[video][p]["speed"] * (relative_frame_id - video_speed[video][p]["start"]))
            else:
                frame_id += int(video_speed[video][p]["speed"] * (video_speed[video][p+1]["start"] - video_speed[video][p]["start"]))
    else:
        frame_id = int(video_speed[video] * relative_frame_id)
    
    frame_id += video_segment_start_frame[video]

    return frame_id
    

# Create a dataset for a video file
def create_dataset(video, dataset_csv):
    dataset_list = []
    dataset_list_u = []

    global_video = video.split("-")[0]

    # Saves the class distribution in the labels
    stat = np.zeros((num_classes))
    stat_count = 0

    if video != "florida-u":
        label_list = os.path.join("masks", video)
        for filename in os.listdir(label_list):
            label_file = os.path.join(label_list, filename)

            i = int(filename.split(".")[0])

            frame_id = get_global_frame_id(video, i)

            if not os.path.exists(os.path.join("frames", global_video, "images", str(frame_id) + ".jpg")):
                raise Exception("Didn't find frame with frame_id=" + str(frame_id))
            
            dataset_list.append((label_file, global_video, str(frame_id)))
            
            frame_path = os.path.join("frames", global_video, "images", str(frame_id) + ".jpg")
            timecode = str(frame_id//25//60).zfill(2) + ":" + str((frame_id//25)%60).zfill(2) + "." + str(int((frame_id%25)/25*100)).zfill(2)
            dataset_csv.loc[len(dataset_csv)] = [label_file, video, i, global_video, frame_id, timecode, frame_path]
            
            # Load label and compute class distribution
            label = io.imread(label_file)
            values, counts = np.unique(label, return_counts=True)
            amount = np.zeros((num_classes))
            amount[values] = counts
            stat += amount
            stat_count += label.size
                
    
    for i in video_unsupervised_index[video]:
        frame_id = get_global_frame_id(video, i)
        dataset_list_u.append(("invalid", global_video, str(frame_id)))

    return dataset_list, dataset_list_u, stat, stat_count

# Writes a list of items to a txt file
def write_to_file(filename, data):
    with open(filename, "w") as file:
        string = ""
        for i in data:
            if type(i) == str:
                string += i + "\n"
            else:
                string += " ".join(i) + "\n"
        
        file.write(string)

# Dataset variants
# Default variant for training
variants = {
    "all": {
        "videos": {
            "florida-01": "test", 
            "florida-02": "train", 
            "florida-03": "val",
            "florida-04": "train",
            "florida-05": "train",
            "florida-06": "train",
            "florida-07": "train",
            "florida-08": "train",
            "florida-09": "train",
            "texas-01": "test2",
            "florida-u": "train",
        },
    },
}

# For each dataset variant
for variant in variants.keys():
    list_train = []
    list_val = []
    list_test1 = []
    list_test2 = []
    list_u = []
    global_amount = np.zeros((num_classes))
    global_count = 0
    dataset_csv = pandas.DataFrame(columns=["label_path", "video_segment", "label_id", "video", "frame_id", "timecode", "frame_path"])

    print("#####", variant, "#####")
    print("")

    # For each video in the dataset variant
    for video, split in variants[variant]["videos"].items():            
        dataset, dataset_u, stat, stat_count = create_dataset(video, dataset_csv)
        
        # Save video file to the relevant dataset
        if split == "val":
            list_val += dataset    
        elif split == "test":
            list_test1 += dataset   
        elif split == "test2":
            list_test2 += dataset
        elif split == "valtest":
            list_val += dataset
            list_test1 += dataset
        else:
            list_train += dataset

        if split not in ["val", "test", "test2", "valtest"]:
            list_u += dataset_u

        # Compute and print the class distribution
        global_amount += stat
        global_count += stat_count

        if stat_count != 0:
            local_stat = stat / stat_count

            print(video + ": " + str(len(dataset)) + " | " + str(len(dataset_u)), ["%.4f" % x for x in local_stat])

    global_stat = global_amount / global_count

    # Print stats
    print("Global: ", ["%.4f" % x for x in global_stat])
    print("")
    print("Train:", len(list_train))
    print("Val:", len(list_val))
    print("Test [Florida]:", len(list_test1))
    print("Test [Texas]:", len(list_test2))
    print("Train Unsupervised:", len(list_u))

    # Write dataset lists to disk
    dataset_csv.to_csv(os.path.join("list",variant,"dataset.csv"), index=False)
    os.makedirs(os.path.join("list",variant), exist_ok=True)
    write_to_file(os.path.join("list",variant,"train.txt"), list_train)
    write_to_file(os.path.join("list",variant,"val.txt"), list_val)
    write_to_file(os.path.join("list",variant,"test.txt"), list_test1)
    write_to_file(os.path.join("list",variant,"test2.txt"), list_test2)
    write_to_file(os.path.join("list",variant,"train_u.txt"), list_u)
    
    print("")
    print("")
    print("")

