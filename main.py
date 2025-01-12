import gradio as gr
import numpy as np
import cv2
import db
from aiotimers import Timer
import pytesseract
from ultralytics import YOLO

timeout = False
timer = None

def timeout_callback():
    global timeout
    timeout = True

def create_timer():
    return Timer(30.00, timeout_callback)

img_streaming = False
stage = 0  # 0 = choose drink, 1 = check pay method, 2 = pay
coupon_result = 0
textbox_text = ""

def check_coupon(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ocr_result = pytesseract.image_to_string(rgb, config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789')
    if db.check_coupon(ocr_result):
        return int(ocr_result)
    else:
        return -1

model = YOLO("best.pt")

def get_name(frame):
    results = model(frame)
    # get max precision result
    max_precision = 0
    max_index = 0
    for i, result in enumerate(results):
        if result['confidence'] > max_precision:
            max_precision = result['confidence']
            max_index = i
    if max_precision > 0.5:
        return results[max_index]['name']
    else:
        return ""

def img_callback(frame):
    global img_streaming, stage, textbox_text, coupon_result, timeout, timer
    if stage == 0 or not img_streaming:
        return np.zeros(frame.shape), textbox_text
    height = frame.shape[0]
    width = frame.shape[1]
    if stage == 1:
        top = height // 4
        left = width // 4
        bottom = height // 4 * 3
        right = width // 4 * 3
        if timeout:
            textbox_text = "扫描超时，请重新扫描"
            img_streaming = False
            timeout = False
            stage = 0
            coupon_result = -1
            return np.zeros(frame.shape), textbox_text
        coupon_result = check_coupon(frame)
        if coupon_result > 0:
            timer.cancel()
            textbox_text = "优惠券可用，编号：" + str(coupon_result)
            img_streaming = False
        elif coupon_result == -1:
            timer.cancel()
            textbox_text = "优惠券不可用"
            img_streaming = False
        cv2.rectangle(frame, (top, left), (right, bottom), (0,0,255), 5)
        return frame, textbox_text
    if stage == 2:
        if timeout:
            textbox_text = "支付超时，请重新支付"
            img_streaming = False
            timeout = False
            stage = 0
            return np.zeros(frame.shape), textbox_text
        result = get_name(frame)
        if result != "":
            timer.cancel()
            if db.check_person(result):
                if db.get_account_money(result) >= (2.4 if coupon_result > 0 else 3):
                    money_after = db.get_account_money(result) - (2.4 if coupon_result > 0 else 3)
                    db.set_account_money(result, money_after)
                    textbox_text = "支付成功，余额：" + str(money_after)
                    img_streaming = False
                    stage = 0
                    coupon_result = 0
                    return np.zeros(frame.shape), textbox_text
                else:
                    textbox_text = "余额不足，请充值"
                    img_streaming = False
                    stage = 0
                    coupon_result = 0
                    return np.zeros(frame.shape), textbox_text
            else:
                textbox_text = "您没有账号，无法支付"
            img_streaming = False
            stage = 0
        return frame, textbox_text

def confirm_callback(drop_value):
    global stage, textbox_text, img_streaming, coupon_result, timer
    if stage == 0:
        stage == 1
        textbox_text = "你选择了" + drop_value + ", 请点击\"确认\"扫描优惠券"
        return textbox_text
    if stage == 1:
        if coupon_result == 0:
            img_streaming = True
            timer = create_timer()
        if coupon_result > 0:
            stage = 2
            textbox_text = "请支付" + "2.4" if coupon_result > 0 else "3" + "元"
        if coupon_result == -1:
            coupon_result = 0
            img_streaming = True
            timer = create_timer()
        return textbox_text
    if stage == 2:
        return textbox_text

with gr.Blocks() as demo:
    with gr.Column():
        dropdown = gr.Dropdown(["可乐", "雪碧", "芬达"], label="请选择饮料", interactive=True, type="value")
        confirm = gr.Button("确认")
    with gr.Column():
        input_img = gr.Image(sources=["webcam"], type="numpy", streaming=True)
        output_box = gr.Textbox(interactive=False)
        input_img.stream(img_callback, inputs=[input_img], outputs=[input_img, output_box], every=0.1)
    
    confirm.click(confirm_callback, inputs=[dropdown], outputs=[output_box])

if __name__ == "__main__":
    demo.launch()