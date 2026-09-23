import zipfile
import os
import codecs
import threading

#Create a destination folder if not existed to save new files with splitted sentences and proper filenames
des_folder = "Country Corpus"
os.makedirs(des_folder, exist_ok=True)

#Second load the zip file which contains the individual corpus files
for file in os.listdir(os.path.join(".", "GlowBe_Data")):
    if file.endswith(".zip"):
        country = file.split("_")[1]

        #Third go through each corpus file within the zip
        with zipfile.ZipFile(os.path.join(".", "GlowBe_Data", file), "r") as zf:
            for name in zf.namelist():
                if name.endswith(".txt"):

                    path = zipfile.Path(zf, at=name)
                    print(name)
                    #Fourth go through each document in the current corpus file
                    with path.open("r", encoding = "utf-8") as f:
                        for line in f:

                            #Some lines are blank
                            if len(line) > 10:

                                #Get the index number
                                text_id = line.split()[0].replace("#","")
                                
                                #Now get the text (with the tags)
                                text = line[2:].replace(text_id, "")
                                
                                #Clean text
                                text = text.strip().replace("<p>","").replace("\n","").replace("\r","").replace("@","").replace(".","\n")

                                #Have   all data
                                write_name = os.path.join(".", des_folder, str(text_id)+"." + country + ".txt")

                                with codecs.open(write_name, "w", encoding = "utf-8") as fw:
                                        if len(text) > 5:
                                            fw.write(text)
 