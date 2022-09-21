import os
import cv2
import mediapipe as mp
import numpy as np
import time
import gaitevents

def process(walk_direction: gaitevents.Side, raw_video_file: str, save_dir: str = ".", use_heavy=True, patientinfo = dict(), rotate_cw=False):

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    if use_heavy:
        complexity = 2
    else:
        complexity = 1

    def match_landmark_to_joint(joint : gaitevents.Joint):
        if joint == gaitevents.Joint.LEFT_TOES:
            return mp_pose.PoseLandmark.LEFT_FOOT_INDEX
        elif joint == gaitevents.Joint.RIGHT_TOES:
            return mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
        elif joint == gaitevents.Joint.LEFT_HIP:
            return mp_pose.PoseLandmark.LEFT_HIP
        elif joint == gaitevents.Joint.RIGHT_HIP:
            return mp_pose.PoseLandmark.RIGHT_HIP
        elif joint == gaitevents.Joint.LEFT_ANKLE:
            return mp_pose.PoseLandmark.LEFT_ANKLE
        elif joint == gaitevents.Joint.RIGHT_ANKLE:
            return mp_pose.PoseLandmark.RIGHT_ANKLE
        elif joint == gaitevents.Joint.LEFT_HEEL:
            return mp_pose.PoseLandmark.LEFT_HEEL
        elif joint == gaitevents.Joint.RIGHT_HEEL:
            return mp_pose.PoseLandmark.RIGHT_HEEL
        elif joint == gaitevents.Joint.LEFT_KNEE:
            return mp_pose.PoseLandmark.LEFT_KNEE
        elif joint == gaitevents.Joint.RIGHT_KNEE:
            return mp_pose.PoseLandmark.RIGHT_KNEE


    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=complexity,
        enable_segmentation=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as pose:

        os.chdir(os.path.dirname(raw_video_file))

        video_capture = cv2.VideoCapture(raw_video_file)
        fps = video_capture.get(5)
        print("Video info:")
        print("FPS: "+str(video_capture.get(5))+" Width: "+str(video_capture.get(3))+" Height: "+str(video_capture.get(4)))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if use_heavy:
            save_filename = "mediapipeheavy_0.avi"
        else:
            save_filename = "mediapipe_0.avi"

        video_writer = cv2.VideoWriter(save_dir+"\\"+save_filename, fourcc, round(fps), (int(video_capture.get(3)), int(video_capture.get(4))))

        time_start = time.time()

        landmark_lists = dict()
        world_landmark_lists = dict()
        for landmark in mp_pose.PoseLandmark:
            landmark_lists[landmark] = []
            world_landmark_lists[landmark] = []

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            elif rotate_cw:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
            frame_height, frame_width, _ = frame.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                print("not detected, using previous frame's data...")
                for landmark in mp_pose.PoseLandmark:
                    if np.shape(landmark_lists[landmark])[0] == 0:
                        landmark_lists[landmark].append([np.nan, np.nan, np.nan])
                        world_landmark_lists[landmark].append([np.nan, np.nan, np.nan])
                    else:
                        landmark_lists[landmark].append(landmark_lists[landmark][-1])
                        world_landmark_lists[landmark].append(world_landmark_lists[landmark][-1])
            else:
                for landmark in mp_pose.PoseLandmark:
                    x = results.pose_landmarks.landmark[landmark].x * frame_width
                    y = results.pose_landmarks.landmark[landmark].y * frame_height
                    z = results.pose_landmarks.landmark[
                            landmark].z * frame_width  # stated on mediapipe website that z follows roughly same scale as x
                    y = -y + int(video_capture.get(4))  # flip and translate y values
                    z = -z # change to right-handed axes convention
                    # if landmark == mp_pose.PoseLandmark.LEFT_KNEE:
                    #     print(results.pose_landmarks.landmark[landmark])
                    landmark_lists[landmark].append([x, y, z])

                    x = results.pose_world_landmarks.landmark[landmark].x
                    y = -results.pose_world_landmarks.landmark[landmark].y
                    z = results.pose_world_landmarks.landmark[landmark].z
                    z = -z  # change to right-handed axes convention
                    world_landmark_lists[landmark].append([x, y, z])
                    # if landmark == mp_pose.PoseLandmark.LEFT_KNEE:
                    #     print(results.pose_world_landmarks.landmark[landmark])

            annotated_image = frame.copy()
            # Draw pose landmarks on the image.
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            #cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
            # Plot pose world landmarks.
            #mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow("Mediapipe - " + raw_video_file, annotated_image)
            video_writer.write(annotated_image)
            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()

        print("Mediapipe processing complete. Time taken: " + str(
            time.time() - time_start) + "s. Average frames processed per second: " + str(
            video_capture.get(cv2.CAP_PROP_FRAME_COUNT) / (time.time() - time_start)) + ".")

        video_capture.release()
        video_writer.release()



        if use_heavy:
            gaitdata = gaitevents.GaitData("mediapipeheavy", fps)
        else:
            gaitdata = gaitevents.GaitData("mediapipe", fps)

        gaitdata.walk_direction = walk_direction
        gaitdata.patientinfo = patientinfo

        gaitdata.data[gaitevents.Joint.LEFT_HEEL] = np.array(landmark_lists[mp_pose.PoseLandmark.LEFT_HEEL])[:, 0:3]
        gaitdata.data[gaitevents.Joint.RIGHT_HEEL] = np.array(landmark_lists[mp_pose.PoseLandmark.RIGHT_HEEL])[:, 0:3]
        gaitdata.data[gaitevents.Joint.LEFT_TOE_BIG] = np.array(landmark_lists[mp_pose.PoseLandmark.LEFT_FOOT_INDEX])[:, 0:3]
        gaitdata.data[gaitevents.Joint.RIGHT_TOE_BIG] = np.array(landmark_lists[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX])[:, 0:3]
        gaitdata.data[gaitevents.Joint.LEFT_ANKLE] = np.array(landmark_lists[mp_pose.PoseLandmark.LEFT_ANKLE])[:, 0:3]
        gaitdata.data[gaitevents.Joint.RIGHT_ANKLE] = np.array(landmark_lists[mp_pose.PoseLandmark.RIGHT_ANKLE])[:,0:3]
        gaitdata.data[gaitevents.Joint.LEFT_KNEE] = np.array(landmark_lists[mp_pose.PoseLandmark.LEFT_KNEE])[:, 0:3]
        gaitdata.data[gaitevents.Joint.RIGHT_KNEE] = np.array(landmark_lists[mp_pose.PoseLandmark.RIGHT_KNEE])[:, 0:3]
        gaitdata.data[gaitevents.Joint.LEFT_HIP] = np.array(landmark_lists[mp_pose.PoseLandmark.LEFT_HIP])[:, 0:3]
        gaitdata.data[gaitevents.Joint.RIGHT_HIP] = np.array(landmark_lists[mp_pose.PoseLandmark.RIGHT_HIP])[:, 0:3]

        midhip = np.array(landmark_lists[mp_pose.PoseLandmark.LEFT_HIP])[:, 0:3] + 0.5 * (
                    np.array(landmark_lists[mp_pose.PoseLandmark.RIGHT_HIP])[:, 0:3] - np.array(
                landmark_lists[mp_pose.PoseLandmark.LEFT_HIP])[:, 0:3])
        gaitdata.data[gaitevents.Joint.MIDDLE_HIP] = midhip

        mid_shoulder = np.array(landmark_lists[mp_pose.PoseLandmark.LEFT_SHOULDER])[:, 0:3] + 0.5 * (
                    np.array(landmark_lists[mp_pose.PoseLandmark.RIGHT_SHOULDER])[:, 0:3] - np.array(
                landmark_lists[mp_pose.PoseLandmark.LEFT_SHOULDER])[:, 0:3])
        gaitdata.data[gaitevents.Joint.MID_SHOULDER] = mid_shoulder

        gaitdata.data[gaitevents.Joint.LEFT_TOES] = gaitdata.data[gaitevents.Joint.LEFT_TOE_BIG]
        gaitdata.data[gaitevents.Joint.RIGHT_TOES] = gaitdata.data[gaitevents.Joint.RIGHT_TOE_BIG]

        gaitdata.data_world[gaitevents.Joint.LEFT_HEEL] = np.array(world_landmark_lists[mp_pose.PoseLandmark.LEFT_HEEL])[:, 0:3]
        gaitdata.data_world[gaitevents.Joint.RIGHT_HEEL] = np.array(world_landmark_lists[mp_pose.PoseLandmark.RIGHT_HEEL])[:, 0:3]
        gaitdata.data_world[gaitevents.Joint.LEFT_TOE_BIG] = np.array(world_landmark_lists[mp_pose.PoseLandmark.LEFT_FOOT_INDEX])[:, 0:3]
        gaitdata.data_world[gaitevents.Joint.RIGHT_TOE_BIG] = np.array(world_landmark_lists[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX])[:, 0:3]
        gaitdata.data_world[gaitevents.Joint.LEFT_ANKLE] = np.array(world_landmark_lists[mp_pose.PoseLandmark.LEFT_ANKLE])[:, 0:3]
        gaitdata.data_world[gaitevents.Joint.RIGHT_ANKLE] = np.array(world_landmark_lists[mp_pose.PoseLandmark.RIGHT_ANKLE])[:,0:3]
        gaitdata.data_world[gaitevents.Joint.LEFT_KNEE] = np.array(world_landmark_lists[mp_pose.PoseLandmark.LEFT_KNEE])[:, 0:3]
        gaitdata.data_world[gaitevents.Joint.RIGHT_KNEE] = np.array(world_landmark_lists[mp_pose.PoseLandmark.RIGHT_KNEE])[:, 0:3]
        gaitdata.data_world[gaitevents.Joint.LEFT_HIP] = np.array(world_landmark_lists[mp_pose.PoseLandmark.LEFT_HIP])[:, 0:3]
        gaitdata.data_world[gaitevents.Joint.RIGHT_HIP] = np.array(world_landmark_lists[mp_pose.PoseLandmark.RIGHT_HIP])[:, 0:3]

        midhip = np.array(world_landmark_lists[mp_pose.PoseLandmark.LEFT_HIP])[:, 0:3] + 0.5 * (
                    np.array(world_landmark_lists[mp_pose.PoseLandmark.RIGHT_HIP])[:, 0:3] - np.array(
                world_landmark_lists[mp_pose.PoseLandmark.LEFT_HIP])[:, 0:3])
        gaitdata.data_world[gaitevents.Joint.MIDDLE_HIP] = midhip

        mid_shoulder = np.array(world_landmark_lists[mp_pose.PoseLandmark.LEFT_SHOULDER])[:, 0:3] + 0.5 * (
                np.array(world_landmark_lists[mp_pose.PoseLandmark.RIGHT_SHOULDER])[:, 0:3] - np.array(
            world_landmark_lists[mp_pose.PoseLandmark.LEFT_SHOULDER])[:, 0:3])
        gaitdata.data_world[gaitevents.Joint.MID_SHOULDER] = mid_shoulder

        gaitdata.data_world[gaitevents.Joint.LEFT_TOES] = gaitdata.data_world[gaitevents.Joint.LEFT_TOE_BIG]
        gaitdata.data_world[gaitevents.Joint.RIGHT_TOES] = gaitdata.data_world[gaitevents.Joint.RIGHT_TOE_BIG]

        if use_heavy:
            gaitdata_filename = "gaitdata_mediapipeheavy.pkl"
        else:
            gaitdata_filename = "gaitdata_mediapipe.pkl"

        gaitevents.save_object(gaitdata, save_dir + "\\"+gaitdata_filename)
        print("Data saved to "+save_dir+"\\"+gaitdata_filename)
