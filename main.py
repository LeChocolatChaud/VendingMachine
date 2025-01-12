import gradio as gr
import numpy as np
import cv2

img_streaming = False
stage = 0  # 0 = choose drink, 1 = check pay method, 2 = pay
coupon_result = -1

def check_coupon(frame):
    # TODO:
    # 1. get ROI (bottom right corner)
    # 2. recognize numbers
    # 3. check if the number is in db
    # 4. check if the coupon has been used
    # 5. return accordingly
    pass

def check_account(frame):
    # TODO:
    # 1. use yolo to recognize people
    # 2. check if the person has an account
    # 3. check if the account has enough money
    # 4. return accordingly
    pass

def img_callback(frame):
    if stage == 0 or not img_streaming:
        return np.zeros(frame.shape)
    height = frame.shape[0]
    width = frame.shape[1]
    if stage == 1:
        top = height // 4
        left = width // 4
        bottom = height // 4 * 3
        right = width // 4 * 3
        coupon_result = check_coupon(frame)
        cv2.rectangle(frame, (top, left), (right, bottom), (0,0,255), 5)
        return frame
    if stage == 2:
        check_account()

with gr.Blocks() as demo:
    with gr.Column():
        dropdown = gr.Dropdown(["可乐", "雪碧", "芬达"], label="请选择饮料", interactive=True)
        
    with gr.Column():
        input_img = gr.Image(sources=["webcam"], type="numpy", streaming=True)

