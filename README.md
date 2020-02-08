Important files to train:

time_maps_flow.py predicts time maps using 1 RGB frame and 3 Optical flow frames --- (Chinmaya, Vulcan) - High priority
time_maps.py predicts time maps using 1 RGB frame --- (Chinmaya) - Highest priority

next_active_object.py predicts the next active object using 1 RGB frame --- (Eadom)
next_active_object_flow.py predicts the next active object using 1 RGB frame and 3 frames of optical flow ---(Chinmaya, Vulcan) - Lower priority
next_active_object_time_maps.py predicts the next active object using 1 RGB frame and 4 frames of time maps --- (Eadom)
next_active_object_multiple_frames.py predicts the next active object using 3 RGB frames --- (Chinmaya, Vulcan) - Lower priority

Command to train file:
python3 -u <file>.py >> log.txt

Notes:

Flow-based methods take a LONG time to run. This is because flow is computed in the dataloader instead of being saved and loaded as the other images.
multiple_frames.py also takes a long time to run.
Make sure you have 20GB of free storage. We're saving weights and data.
