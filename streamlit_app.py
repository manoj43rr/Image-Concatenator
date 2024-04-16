import streamlit as st
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_elements_2d_array(arr):
    # Extracting elements
    first_row = arr[0]
    first_column = [row[0] for row in arr]
    last_row = arr[-1]
    last_column = [row[-1] for row in arr]

    return first_row.tolist(), first_column, last_row.tolist(), last_column

def operation_row(arr1, arr2):
    diff1 = [len(arr1[abs(arr1['row1'] - arr2['row1']) < 20])]
    diff2 = [len(arr1[abs(arr1['rowz'] - arr2['rowz']) < 20])]
    diff3 = [len(arr1[abs(arr1['row1'] - arr2['rowz']) < 20])]
    return diff1, diff2, diff3

def operation_col(arr1, arr2):
    diff1 = [len(arr1[abs(arr1['col1'] - arr2['col1']) < 20])]
    diff2 = [len(arr1[abs(arr1['colz'] - arr2['colz']) < 20])]
    diff3 = [len(arr1[abs(arr1['col1'] - arr2['colz']) < 20])]
    return diff1, diff2, diff3

def main():
    st.title("Image Concatenation App")
    
    uploaded_files = st.file_uploader("Upload two images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files is not None and len(uploaded_files) == 2:
        img1 = cv.imdecode(np.fromstring(uploaded_files[0].read(), np.uint8), cv.IMREAD_COLOR)
        img2 = cv.imdecode(np.fromstring(uploaded_files[1].read(), np.uint8), cv.IMREAD_COLOR)

        st.subheader("Uploaded Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1, channels="BGR", use_column_width=True)
        with col2:
            st.image(img2, channels="BGR", use_column_width=True)

        gray1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

        if gray1.shape == gray2.shape:
            if gray1.shape[0] < gray1.shape[1]:
                gray1 = np.rot90(gray1, 1)
                gray2 = np.rot90(gray2, 1)
        elif gray1.shape[0] == gray2.shape[1] and gray2.shape[0] == gray1.shape[1]:
            if gray1.shape[0] > gray2.shape[0]:
                gray2 = np.rot90(gray2, 3)
            elif gray2.shape[0] > gray1.shape[0]:
                gray1 = np.rot90(gray1, 3)

        p1r1, p1c1, p1r2, p1c2 = extract_elements_2d_array(gray1)
        p2r1, p2c1, p2r2, p2c2 = extract_elements_2d_array(gray2)

        pic1_rows = pd.DataFrame({'row1': p1r1, 'rowz': p1r2})
        pic1_cols = pd.DataFrame({'col1': p1c1, 'colz': p1c2})

        pic2_rows = pd.DataFrame({'row1': p2r1, 'rowz': p2r2})
        pic2_cols = pd.DataFrame({'col1': p2c1, 'colz': p2c2})

        a, b, c = operation_row(pic1_rows, pic2_rows)
        data = pd.DataFrame({'r1r1': a, 'rzrz': b, 'r1rz': c})
        a, b, c = operation_row(pic2_rows, pic1_rows)
        data = data.append({'r1r1': a[0], 'rzrz': b[0], 'r1rz': c[0]}, ignore_index=True)

        a, b, c = operation_col(pic1_cols, pic2_cols)
        data1 = pd.DataFrame({'c1c1': a, 'czcz': b, 'c1cz': c})
        a, b, c = operation_col(pic2_cols, pic1_cols)
        data1 = data1.append({'c1c1': a[0], 'czcz': b[0], 'c1cz': c[0]}, ignore_index=True)

        if max(data1['c1cz']) > max(data['r1rz']):
            if max(data['r1r1']) > max(data['r1rz']) and data1['c1cz'][0] < data1['c1cz'][1]:
                concatenated = np.concatenate((gray1, gray2), axis=1)
            else:
                gray1_rot = np.rot90(gray1, 2)
                gray2_rot = np.rot90(gray2, 2)
                concatenated = np.concatenate((gray1_rot, gray2_rot), axis=1)
        else:
            if max(data1['czcz']) > max(data1['c1c1']) and max(data1['czcz']) > max(data1['c1cz']):
                rot = np.rot90(gray2, 2)
                concatenated = np.concatenate((gray1, rot), axis=1)
            elif max(data1['c1c1']) > max(data1['czcz']) and max(data1['c1c1']) > max(data1['c1cz']):
                rot = np.rot90(gray1, 2)
                concatenated = np.concatenate((rot, gray2), axis=1)

        gray_image = concatenated

        color_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)

        # Assign the intensity values from the grayscale image to all three channels
        color_image[:, :, 0] = gray_image
        color_image[:, :, 1] = gray_image
        color_image[:, :, 2] = gray_image

        st.subheader("Concatenated Image")
        st.image(color_image, channels="BGR", use_column_width=False, width=800)

if __name__ == "__main__":
    main()
