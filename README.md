Important files to train:

time_maps_flow.py predicts time maps using 1 RGB frame and 3 Optical flow frames --- Medium Priority
time_maps.py predicts time maps using 1 RGB frame --- High priority

next_active_object.py predicts the next active object using 1 RGB frame --- High Priority
next_active_object_flow.py predicts the next active object using 1 RGB frame and 3 frames of optical flow --- Lower priority
next_active_object_time_maps.py predicts the next active object using 1 RGB frame and 4 frames of time maps --- High Priority
next_active_object_multiple_frames.py predicts the next active object using 3 RGB frames ---Lower priority

Command to train file:
python3 -u <file>.py >> log.txt

Notes:

Make sure you have 50GB of storage. We're saving multiple weights for each training session

Optical_flow.py is responsible for generating the optical flow frames
