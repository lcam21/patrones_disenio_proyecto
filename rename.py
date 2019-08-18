# Pythono3 code to rename multiple  
# files in a directory or folder 
  
# importing os module 
import os 
  
i = 0

folder = "train" 
for filename in os.listdir(folder):
	dst = filename[37:39] + "_test" + str(i) + ".png"
	src =folder + '/' + filename 
	dst =folder + '/' + dst 
	os.rename(src, dst) 
	i += 1
  
