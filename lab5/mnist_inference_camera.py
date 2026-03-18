import sys
import argparse
import os
from fnmatch import fnmatch

from jetson_inference import imageNet
from jetson_utils import *
import numpy as np


def crop(img, crop_roi):
	crop_img = cudaAllocMapped(width=crop_roi[2]-crop_roi[0], height=crop_roi[3]-crop_roi[1], format=img.format)
	cudaCrop(img, crop_img, crop_roi)
	return crop_img

def resize(img, new_h_w):
	resized_img = cudaAllocMapped(width=new_h_w[1],height=new_h_w[0],format=img.format)
	cudaResize(img, resized_img)
	return resized_img

def concat_imgs(img_A, img_B):
    imgOutput = cudaAllocMapped(width=img_A.width + img_B.width, 
                                height=max(img_A.height, img_B.height),
                                format=img_A.format)

    cudaOverlay(img_A, imgOutput, 0, 0)
    cudaOverlay(img_B, imgOutput, img_A.width, img_A.height-img_B.height)
    return imgOutput

def convert_color(img, output_format):
	converted_img = cudaAllocMapped(width=img.width, height=img.height, format=output_format)
	cudaConvertColor(img, converted_img)
	return converted_img


# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=imageNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="csi://0", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="rtp://192.168.55.100:8554", nargs='?', help="URI of the output stream")
parser.add_argument("--model-dir", type=str, default=".", help="Path of model to load")
parser.add_argument("--model", type=str, default="", help="Path of model to load")
parser.add_argument("--labels", type=str, default="", help="Path to labels text file.")
parser.add_argument("--topK", type=int, default=3, help="show the topK number of class predictions (default: 3)")

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

if args.model =="": #.contains('*'):
    for file in os.listdir(args.model_dir):
        if fnmatch(file, '*.onnx'):
            print("No --model option provided, auto-selecting:", file)
            args.model = file
            break

model_path = os.path.join(args.model_dir,args.model)
labels_path = os.path.join(args.model_dir,"labels.txt") if args.labels == "" else args.labels

print("Loading weights from:", model_path)
print("Loading labels from:", labels_path)

net = imageNet("", 
                model=model_path, labels=labels_path,
                input_blob="input_0", output_blob="output_0")

camera = videoSource(args.input)

display_size = 512
roi_size = 28*10
input_size = 28
display_top = 720//2 - display_size//2
display_left = 1080//2 - display_size//2
display_bounds = (display_left,display_top,display_left+display_size,display_top+display_size)
input_display_scale = 10
# display_bounds = (280,0,1000,720)
roi_top_left = display_size//2 - roi_size//2
roi_bounds = (roi_top_left,roi_top_left,roi_top_left+roi_size,roi_top_left+roi_size)


display = videoOutput(args.output,argv=[f'--width={display_size+input_size+input_size*input_display_scale} --height=512'])
font = cudaFont()

bar_len = 18
while True:
    
    img = camera.Capture()
    if img is None: # capture timeout
        continue

    img = convert_color(img, 'rgb32f')
    
    imgDisplay = crop(img, display_bounds) 
    imgROI = crop(imgDisplay, roi_bounds)
    imgInput = resize(imgROI, (input_size,input_size))
    # saveImage("/custom_classifier/out.jpg", imgInput)
    imgInputUpscaled = resize(imgROI, (input_size,input_size))
    cudaDrawRect(imgDisplay, roi_bounds, (255,127,0,128))
    imgDisplay = concat_imgs(imgDisplay, imgInput)

    imgInputUpscaled_np = cudaToNumpy(imgInputUpscaled)
    imgInputUpscaled_np = np.repeat(imgInputUpscaled_np, input_display_scale, axis=0)
    imgInputUpscaled_np = np.repeat(imgInputUpscaled_np, input_display_scale, axis=1)
    imgInputUpscaled = cudaFromNumpy(imgInputUpscaled_np)
    imgDisplay = concat_imgs(imgDisplay, imgInputUpscaled)

    predictions = net.Classify(imgInput, topK=args.topK)

    # draw predicted class labels
    text = ""
    for n, (classID, confidence) in enumerate(predictions):
        classLabel = net.GetClassLabel(classID)

        bar = "|"+"#"*int(np.round(bar_len*confidence))+ " "*int(np.round(bar_len*(1-confidence))) + "|"
        new_line = f"{classID}  {(confidence*100.0):05.2f}% {bar}\n"
        text += new_line

        font.OverlayText(imgDisplay, text=new_line, 
                        x=5, y=5 + n * (font.GetSize() + 5),
                        color=font.White, background=font.Gray40)
    fps_text = f"FPS: {net.GetNetworkFPS():.1f}"
    font.OverlayText(imgDisplay, text=fps_text, 
                        x=600, y=5,
                        color=font.White, background=font.Gray40)
    text = fps_text+"\n"+text             
    print(text)

    display.Render(imgDisplay)
    
