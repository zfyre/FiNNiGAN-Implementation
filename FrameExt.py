import os
import cv2


VidDirectory = os.path.join('data','Videos') # --> os.path.join('data','Videos')
# Will be the absolute path:
"""Change This Absolute Path!!"""
Abs_DatasetPath = "---" # --> "C:/.../FiniGAN/data/output/" 
#Will be a relative path:
Rel_DatasetPath = os.path.join('data','output') # --> os.path.join('data','output')

directory = os.fsencode(VidDirectory)


def saveFrame(success,img,dirname,name):
    if success:
        result=cv2.imwrite(Abs_DatasetPath+dirname+'/'+name,img)
        print(Abs_DatasetPath+dirname+'/'+name)
    if result==True:
        print('File saved successfully')
    else:
        print('Error in saving file')


cnt = -1
def VidCap(VidPath,count):
    cap = cv2.VideoCapture(VidPath)

    for frame_idx in range((int(cap.get(cv2.CAP_PROP_FRAME_COUNT))//3)*3):
        # Read Frame:
        success,frame = cap.read()
        global cnt 
        if((frame_idx%3)==0):
            cnt += 1
            dir_path = os.path.join(Rel_DatasetPath,'data'+str(cnt))
            
            # if os.path.exists(dir_path): # Set-ExecutionPolicy Unrestricted
            #     os.remove(dir_path) 
            """Above Removal Not working, reason is not clear"""
            
            # Create a directory as data_i: 
            os.mkdir(dir_path) 
            # Add current frame also to the directory
            saveFrame(success,frame,'data'+str(cnt),'frame1.png')

        elif((frame_idx%3)==1):
            # Add current Frame to the directory
            saveFrame(success,frame,'data'+str(cnt),'frame2.png')

        else:
            # Add current Frame to the directory
            saveFrame(success,frame,'data'+str(cnt),'frame3.png')

        # Rendering The Processing Video
        cv2.imshow('Video Player',frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Close down Everything and release Video Writer:
    cap.release()
    cv2.destroyAllWindows()



# Iterating The Videos containing Directory
outcount = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    file_path = os.path.join(VidDirectory,filename)
    VidCap(file_path,outcount)
    print(filename)
    outcount+=1




