import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from collections import defaultdict
import pandas as pd
import tempfile
import os

# Tải mô hình
model = YOLO("my_model/yolov11-custom3/weights/last.pt")

# Tiêu đề
st.markdown(
    """
    <h1 style='text-align: center; color: purple; font-size: 40px; white-space: nowrap;'>
         NHẬN DIỆN VI KHUẨN BẰNG YOLOV11 
    </h1>
    """,
    unsafe_allow_html=True
)

# Lựa chọn chế độ
option = st.radio("Chọn chế độ nhập liệu", ["Tải ảnh", "Chụp webcam", "Video"])

def hien_thi_ket_qua(image_bgr, results):
    # Vẽ bounding boxes
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            confidence = box.conf[0]
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_bgr, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption='Kết quả nhận diện', use_container_width=True)
    return results

def thong_ke_va_khuyen_nghi(results):
    st.subheader("Thống kê nhận diện:")
    label_data = defaultdict(list)
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            label_data[label].append(confidence)

    summary_data = {
        "Vi khuẩn": [],
        "Số lượng": [],
        "Độ chính xác trung bình (%)": []
    }

    for label, confs in label_data.items():
        summary_data["Vi khuẩn"].append(label)
        summary_data["Số lượng"].append(len(confs))
        summary_data["Độ chính xác trung bình (%)"].append(round(np.mean(confs) * 100, 2))

    df = pd.DataFrame(summary_data)
    st.dataframe(df)

    # Biện pháp phòng chống
    st.subheader("Biện pháp phòng chống:")
    for label in set(label_data.keys()):
        if label == "E-coli":
            st.markdown("""
            - **E-coli**:
                1. Rửa tay kỹ bằng xà phòng sau khi đi vệ sinh và trước khi ăn.
                2. Nấu chín thịt bò và các loại thịt khác đúng cách.
                3. Tránh uống nước chưa đun sôi hoặc chưa lọc sạch.
                4. Làm sạch các dụng cụ nấu ăn, thớt, bề mặt tiếp xúc với thực phẩm sống.
            """)
        elif label == "Paramecium":
            st.markdown("""
            - **Paramecium**:
                1. Sử dụng nguồn nước sạch và đã xử lý khi sinh hoạt.
                2. Lắp đặt hệ thống lọc nước gia đình.
                3. Vệ sinh định kỳ bể chứa nước và thiết bị lọc.
                4. Không tiếp xúc với nước ao hồ ô nhiễm.
            """)
        elif label == "Yeast":
            st.markdown("""
            - **Yeast**:
                1. Duy trì vệ sinh cá nhân, đặc biệt ở các vùng da ẩm ướt.
                2. Tránh mặc đồ bó sát và giữ vùng kín luôn khô ráo.
                3. Hạn chế sử dụng kháng sinh nếu không cần thiết.
                4. Ăn uống cân bằng để giữ hệ vi sinh đường ruột khỏe mạnh.
            """)
        elif label == "bacillus":
            st.markdown("""
            - **bacillus**:
                1. Bảo quản thực phẩm ở nhiệt độ phù hợp.
                2. Không ăn thực phẩm để lâu ngoài môi trường.
                3. Vệ sinh tay và dụng cụ nấu ăn sạch sẽ.
                4. Tránh dùng thực phẩm đã bị biến đổi màu hoặc có mùi lạ.
            """)
        elif label == "Rods":
            st.markdown("""
            - **Rods**:
                1. Thu gom và xử lý rác thải đúng cách.
                2. Khử trùng các khu vực công cộng, nhà vệ sinh thường xuyên.
                3. Không để thực phẩm tiếp xúc với môi trường bẩn.
                4. Dùng nước sạch và thiết bị bảo hộ khi xử lý chất thải.
            """)
        elif label == "coco" or label == "Cocci":
            st.markdown("""
            - **Cocci (vi khuẩn hình cầu)**:
                1. Giữ gìn vệ sinh cá nhân, đặc biệt là rửa tay đúng cách và thường xuyên.
                2. Sát khuẩn vết thương ngay khi bị xây xước để tránh nhiễm trùng.
                3. Không dùng chung đồ dùng cá nhân như khăn mặt, dao cạo, bàn chải đánh răng.
                4. Tăng cường hệ miễn dịch qua chế độ ăn uống lành mạnh và nghỉ ngơi hợp lý.
            """)

#Tai anh len
if option == "Tải ảnh":
    uploaded_file = st.file_uploader("Chọn một ảnh để nhận diện", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        results = model.predict(image_bgr)
        hien_thi_ket_qua(image_bgr, results)
        thong_ke_va_khuyen_nghi(results)

#Anh chup tu webcam
elif option == "Chụp webcam":
   picture = st.camera_input("Chụp ảnh từ webcam") 
   if picture:
        image = Image.open(picture)
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        results = model.predict(image_bgr)
        hien_thi_ket_qua(image_bgr, results)
        thong_ke_va_khuyen_nghi(results)

#Video 
elif option == "Video":
    video_file = st.file_uploader("Tải video để nhận diện" ,type=["mp4", "avi", "mov"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        frame_idx = 0
        all_results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx > 300:
                break
            frame_idx += 1

            results = model.predict(frame)
            all_results.extend(results)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = model.names[int(box.cls[0])]
                    confidence = box.conf[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, caption=f"Frame {frame_idx}",use_container_width=True)

        cap.release()
        thong_ke_va_khuyen_nghi(all_results)
