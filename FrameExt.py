import cv2
import os
from matplotlib import pyplot as plt

# Setup the Capture:
# cap = cv2.VideoCapture(os.path.join('data','Videos','vid0.y4m'))

# Grab a frame:
# success , frame = cap.read()

# Rendering the Frame:
# print(frame.shape)
# plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
# plt.show()

# Release the capture:
# cap.release()
# print(cap.read())

# Capture Properties

# print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# print(cap.get(cv2.CAP_PROP_FPS))
 

# Working with Video Captures:

cap = cv2.VideoCapture(os.path.join('data','Videos','vid0.y4m'))

# Setup Video Writer:
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)

videowriter = cv2.VideoWriter(os.path.join('data','output','output.avi'),cv2.VideoWriter_fourcc('P','I','M','1'),fps,(width,height),isColor = False)

for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    # Read Frame:
    success,frame = cap.read()
    # Transform:
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    # show Image
    cv2.imshow('Video Player',gray)
    # Write out our frame:
    videowriter.write(gray)
    # Render:
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
# Close down Everything:
cap.release()
cv2.destroyAllWindows()
# Relase our video Writer:
videowriter.release()