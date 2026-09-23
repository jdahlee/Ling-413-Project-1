import os

#Set root location
root = os.path.join(".", "Country Corpus")

#Look at each file
count = 0
for file in os.listdir(root):
    if file.endswith(".txt"):
        if count == 0:
            print("beginning script")
        #Get the text id
        text_id = file.split(".")[0]
        country = file.split(".")[1]
        
        #Make dir
        os.makedirs(os.path.join(root, country), exist_ok=True)

        os.rename(os.path.join(root, file), os.path.join(root, country, file))
        if (count % 10000 == 0):
            print(count)
        count += 1