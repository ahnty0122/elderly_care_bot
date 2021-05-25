import argparse
import cv2
import sys
import numpy as np
import time
<<<<<<< HEAD:pose_estimation.py
try:
=======
import RPi.GPIO as GPIO

#GPIO.setmode(GPIO.BCM)
#led
#GPIO.setup(4, GPIO.OUT)
#button
#GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        #breakpoint()
        
        self.stream = cv2.VideoCapture(0)
        print("Camera initiated.")
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected keypoints (specify between 0 and 1).',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
parser.add_argument('--output_path', help="Where to save processed imges from pi.",
                    required=True)

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tensorflow is not installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tensorflow')
if pkg is None:
>>>>>>> 58f68bb781a070f6f7a831ce11a5f280c7456797:rpi_pose_estimation/run_pose_estimation.py
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

<<<<<<< HEAD:pose_estimation.py

def getKeypoints(probMap, threshold=0.1):

    mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []
    contours = None
    try:
        #OpenCV4.x
        contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        #OpenCV3.x
        _, contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


def getValidPairs(outputs, w, h):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7

    for k in range(len(mapIdx)):
        pafA = outputs[0, mapIdx[k][0], :, :]
        pafB = outputs[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (w, h))
        pafB = cv2.resize(pafB, (w, h))

        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            valid_pairs.append(valid_pair)
        else:
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs


def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobilenet_v2_pose_256_256_dm100_integer_quant.tflite", help="Path of the detection model.")
    parser.add_argument("--usbcamno", type=int, default=0, help="USB Camera number.")
    parser.add_argument("--camera_type", default="usb_cam", help="set usb_cam or raspi_cam or video_file")
    parser.add_argument("--camera_width", type=int, default=640, help="width.")
    parser.add_argument("--camera_height", type=int, default=480, help="height.")
    parser.add_argument("--vidfps", type=int, default=150, help="Frame rate.")
    parser.add_argument("--num_threads", type=int, default=4, help="Threads.")
    parser.add_argument("--input_video_file", default='', help="Input video file")
    args = parser.parse_args()

    model = args.model
    usbcamno = args.usbcamno
    camera_type = args.camera_type
    width = args.camera_width
    height = args.camera_height
    vidfps = args.vidfps
    num_threads = args.num_threads

    fps = ""
    framecount = 0
    time1 = 0
    elapsedTime = 0

    keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee',
                        'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
    POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9], [9,10], [1,11],
                [11,12], [12,13], [1,0], [0,14], [14,16], [0,15], [15,17], [2,17], [5,16]]
    mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], [23,24], [25,26],
            [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], [55,56], [37,38], [45,46]]
    colors = [[0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255],
            [0,100,255], [0,255,0], [255,200,100], [255,0,255], [0,255,0],
            [255,200,100], [255,0,255], [0,0,255], [255,0,0], [200,200,0],
            [255,0,0], [200,200,0], [0,0,0]]

    if args.input_video_file != "":
        # WORKAROUND
        print("[Info] --input_video_file has an argument. so --device was replaced to 'video_file'.")
        camera_type = "video_file"

    if camera_type == "usb_cam":
        cam = cv2.VideoCapture(usbcamno)
        cam.set(cv2.CAP_PROP_FPS, vidfps)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    elif camera_type == "raspi_cam":
        from picamera.array import PiRGBArray
        from picamera import PiCamera
        cam = PiCamera()
        cam.resolution = (width, height)
        stream = PiRGBArray(cam)
    elif camera_type == "video_file":
        cam = cv2.VideoCapture(args.input_video_file)

    else:
        print('[Error] --camera_type: wrong device')
        parser.print_help()
        sys.exit()

    cv2.namedWindow('pose estimation pi', cv2.WINDOW_AUTOSIZE)

    interpreter = Interpreter(model_path=model)
    interpreter.allocate_tensors()
    try:
        interpreter.set_num_threads(int(num_threads))
    except:
        print("WARNING: The installed PythonAPI of Tensorflow/Tensorflow Lite runtime does not support Multi-Thread processing.")
        print("WARNING: It works in single thread mode.")
        print("WARNING: If you want to use Multi-Thread to improve performance on aarch64/armv7l platforms, please refer to one of the below to implement a customized Tensorflow/Tensorflow Lite runtime.")
        print("https://github.com/PINTO0309/Tensorflow-bin.git")
        print("https://github.com/PINTO0309/TensorflowLite-bin.git")
        pass
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    h = input_details[0]['shape'][1] #256
    w = input_details[0]['shape'][2] #256

    threshold = 0.25
    nPoints = 18

    try:

        while True:
            t1 = time.perf_counter()

            if camera_type == 'raspi_cam':
                cam.capture(stream, 'bgr', use_video_port=True)
                color_image = stream.array
                stream.truncate(0)
            else:
                ret, color_image = cam.read()
                # color_image = color_image[30:510, 160:800] # resize
                if not ret:
                    continue

            colw = color_image.shape[1]
            colh = color_image.shape[0]
            new_w = int(colw * min(w/colw, h/colh))
            new_h = int(colh * min(w/colw, h/colh))

            resized_image = cv2.resize(color_image, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
            canvas = np.full((h, w, 3), 128)
            canvas[(h - new_h)//2:(h - new_h)//2 + new_h,(w - new_w)//2:(w - new_w)//2 + new_w, :] = resized_image

            prepimg = canvas.astype(np.float32)
            prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
            interpreter.set_tensor(input_details[0]['index'], prepimg)
            interpreter.invoke()
            outputs = interpreter.get_tensor(output_details[0]['index']) #(1, 32, 32, 57)
            outputs = outputs.transpose((0, 3, 1, 2))  # NHWC to NCHW, (1, 57, 32, 32)

            detected_keypoints = []
            keypoints_list = np.zeros((0, 3))
            keypoint_id = 0

            for part in range(nPoints):
                probMap = outputs[0, part, :, :]
                probMap = cv2.resize(probMap, (canvas.shape[1], canvas.shape[0])) # (256, 256)
                keypoints = getKeypoints(probMap, threshold)
                keypoints_with_id = []

                for i in range(len(keypoints)):
                    keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                    keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                    keypoint_id += 1

                detected_keypoints.append(keypoints_with_id)

            frameClone = np.uint8(canvas.copy())
            for i in range(nPoints):
                for j in range(len(detected_keypoints[i])):
                    cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)

            valid_pairs, invalid_pairs = getValidPairs(outputs, w, h)
            personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)

            for i in range(17):
                for n in range(len(personwiseKeypoints)):
                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

            cv2.putText(frameClone, fps, (w-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
            frameClone = cv2.resize(frameClone, (colw, colw))
            cv2.imshow("pose estimation pi" , frameClone)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'): # when ESC key or q key is pressed break
                break

            # FPS calculation
            framecount += 1
            if framecount >= 15:
                fps = "(Playback) {:.1f} FPS".format(time1/15)
                framecount = 0
                time1 = 0
            t2 = time.perf_counter()
            elapsedTime = t2-t1
            time1 += 1/elapsedTime

    except:
        import traceback
        traceback.print_exc()

    finally:

        print("\n\nFinished\n\n")
=======
try:
    print("Progam started - waiting for button push...")
    while True:
    #if True:
        #make sure LED is off and wait for button press
        #if cv2.waitKey(1) == ord('p') or led_on and not GPIO.input(17):
        key = input()
        if key == 'p':
        #if not led_on and  not GPIO.input(17):
        #if True:
            #timestamp an output directory for each capture
            print("start")
            outdir = pathlib.Path(args.output_path) / time.strftime('%Y-%m-%d_%H-%M-%S-%Z')
            outdir.mkdir(parents=True)
            #GPIO.output(4, True)
            time.sleep(.1)
            #led_on = True
            f = []

            # Initialize frame rate calculation
            frame_rate_calc = 1
            freq = cv2.getTickFrequency()
            videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
            time.sleep(1)

            #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

            while True:
            #while cv2.waitkey(33) != ord('q'):
                print('running loop')
                # Start timer (for calculating frame rate)
                t1 = cv2.getTickCount()
                
                # Grab frame from video stream
                frame1 = videostream.read()
                # Acquire frame and resize to expected shape [1xHxWx3]
                frame = frame1.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (width, height))
                input_data = np.expand_dims(frame_resized, axis=0)
                
                frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
                if floating_model:
                    input_data = (np.float32(input_data) - input_mean) / input_std

                # Perform the actual detection by running the model with the image as input
                interpreter.set_tensor(input_details[0]['index'],input_data)
                interpreter.invoke()
                
                #get y,x positions from heatmap
                coords = sigmoid_and_argmax2d(output_details, min_conf_threshold)
                #keep track of keypoints that don't meet threshold
                drop_pts = list(np.unique(np.where(coords ==0)[0]))
                #get offets from postions
                offset_vectors = get_offsets(output_details, coords)
                #use stide to get coordinates in image coordinates
                keypoint_positions = coords * output_stride + offset_vectors
            
                # Loop over all detections and draw detection box if confidence is above minimum threshold
                for i in range(len(keypoint_positions)):
                    #don't draw low confidence points
                    if i in drop_pts:
                        continue
                    # Center coordinates
                    x = int(keypoint_positions[i][1])
                    y = int(keypoint_positions[i][0])
                    center_coordinates = (x, y)
                    radius = 2
                    color = (0, 255, 0)
                    thickness = 2
                    cv2.circle(frame_resized, center_coordinates, radius, color, thickness)
                    if debug:
                        cv2.putText(frame_resized, str(i), (x-4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1) # Draw label text
     
                frame_resized = draw_lines(keypoint_positions, frame_resized, drop_pts)
                
                # Draw framerate in corner of frame - remove for small image display
                #cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
                #cv2.putText(frame_resized,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
                frame_resized = cv2.resize(frame_resized, (640, 480))
                cv2.imshow('img', frame_resized)
                
                # Calculate framerate
                t2 = cv2.getTickCount()
                time1 = (t2-t1)/freq
                frame_rate_calc= 1/time1
                f.append(frame_rate_calc)
    
                #save image with time stamp to directory
                path = str(outdir) + '/'  + str(datetime.datetime.now()) + ".jpg"

                status = cv2.imwrite(path, frame_resized)
                
                # Press 'q' to quit
                if cv2.waitKey(1) == ord('q') or led_on and not GPIO.input(17):
                #if input() == 'q':
                    print(f"Saved images to: {outdir}")
                    #GPIO.output(4, False)
                    #led_on = False
                    # Clean up
                    cv2.destroyAllWindows()
                    videostream.stop()
                    time.sleep(2)
                    break

except KeyboardInterrupt:
    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()
    print('Stopped video stream.')
    #GPIO.output(4, False)
    #GPIO.cleanup()
    #print(str(sum(f)/len(f)))
>>>>>>> 58f68bb781a070f6f7a831ce11a5f280c7456797:rpi_pose_estimation/run_pose_estimation.py
