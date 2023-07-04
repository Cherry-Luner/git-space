import os

folder_path = "./data/PlantVillage/train"
for root,dirname,filename in os.walk(folder_path):
    print(filename)
