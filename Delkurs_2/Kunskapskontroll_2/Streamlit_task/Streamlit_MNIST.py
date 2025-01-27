import cv2
import numpy as np
import pickle
import streamlit as st

# Tester for MNIST using Support Vector Machine (SVM)
# Use camera as input of digits that are predicted using a pre trained SVM model.
# Claes Rolén - 2025-01-25 - First version


@st.cache_data
def getModel():
    print("Loading trained SVM model")
    with open("TrainedSVM_BIG_SVM_PROB.pkl", "rb") as f:
        svm_c = pickle.load(f)
    with open("Trained_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return svm_c, scaler


def main_loop():
    ################################################################
    # Load the trained SVM model and scaler
    svm_c, scaler = getModel()

    ################################################################
    # Setup Streamlit widgets
    st.title("Digit classifier on Video Stream")
    st.write("")
    cam_option = st.selectbox("Which camera to use?", ("Default", "External"))
    run = st.checkbox("Run")
    canny_thr1 = st.sidebar.slider(
        "Canny low threshold", value=120, min_value=1, max_value=255
    )
    canny_thr2 = st.sidebar.slider(
        "Canny high threshold", value=230, min_value=1, max_value=255
    )
    dig_thr = st.sidebar.slider(
        "Current digit black level threshold", value=130, min_value=1, max_value=255
    )
    col11, col12 = st.columns(2)
    with col11:
        st.caption("Input Image")
        INPUT_WINDOW = st.image([])
    with col12:
        st.caption("Canny Image")
        CANNY_WINDOW = st.image([])

    col21, col22 = st.columns(2)
    with col21:
        st.caption("Current digit")
        DIGIT_WINDOW = st.image([])
    with col22:
        st.caption("Current Digit Predictions")
        CHART1 = st.empty()

    ################################################################
    # Setup camera - instance 0 = default (internal) camera
    #                Instance 1 = external camera
    if cam_option == "Default":
        instance = 0
    else:
        instance = 1
    cap = cv2.VideoCapture(instance)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    # Reduce resolution to 320x240 for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    ################################################################
    # Main loop
    while run:
        ret, img0 = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)

        # Edge detection
        canny = cv2.Canny(gray, canny_thr1, canny_thr2, 1)

        # Find contours and extract ROI
        cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)  # Find ROI and draw a rectangle
            cv2.rectangle(img0, (x, y), (x + w, y + h), (0, 250, 250), 1)

            # Extract ROI from gray image and threshold the lower levels
            digN = 255 - gray[y : y + h, x : x + w]
            _, digN = cv2.threshold(digN, dig_thr, 255, type=cv2.THRESH_TOZERO)

            # Fit ROI to 28x28 image
            r = max(w, h)
            bg = np.zeros((r, r), dtype=np.uint8)
            row = max(r // 2 - h // 2, 0)
            col = max(r // 2 - w // 2, 0)
            bg[row : row + h, col : col + w] = digN
            img_sc = cv2.resize(bg, (22, 22))
            img_out = np.zeros((28, 28), dtype=np.uint8)
            img_out[3:25, 3:25] = img_sc

            # Predict
            preds = svm_c.predict_proba(scaler.transform(img_out.reshape(1, -1)))[0]

            # Overlay prediction info, max from probabilities
            cv2.putText(
                img0,
                str(np.argmax(preds)),
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        # Display results
        if len(cnts) > 0:
            # Use another color for top y-position digit (current digit)
            cv2.rectangle(img0, (x, y), (x + w, y + h), (0, 50, 250), 2)
            DIGIT_WINDOW.image(img_out, width=150)
            CHART1.bar_chart(preds, horizontal=True)
        INPUT_WINDOW.image(img0, channels="BGR")
        CANNY_WINDOW.image(canny)
    # else:
    #     st.write("Stopped")

if __name__ == "__main__":
    main_loop()
