import os

currentPath = r'C:\Users\Charles\Documents\Cours\3A\DeepLearning\Projet2\images'
dirList = os.scandir(os.path.join(currentPath,r'images/Albrecht_Durer'))

for image in dirList:
    print(image.name)
    splits=str.split(image.name,'_Durer')
    os.rename(os.path.join(currentPath,r'images/Albrecht_Durer',image.name), os.path.join(currentPath,r'images/Albrecht_Durer2','Albrecht_Durer_'+splits[1]))
