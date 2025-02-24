# Streamlit Demo - Kunskapskontroll 2.1

Here you can try a live stream classification using a pretrained SVM classifier. The input is either the default camera (internal) or an external USB device. 

The camera resolution is limited to 320x240 to lower the computational effort. 
Some image preprocessing is made to localize the digit and to convert it to a suitable format.
 - convert to grayscale format
 - detect edges using Canny edge detector
 - find contour groups
 - extract ROI (region of interest) from the contour list
 - center an scale the ROI to an 28x28 image
 - threshold the image towards zero

 Some parameters are tunable via sliders at the sidebar. Unfortunately when updating the sliders the python script is run from start again (by design in Streamlit), haven't found an easy solution to mitigate that.

 The images shown are from different blocks in the image processing pipe. The red border rectangle shows the *Current Digit*
and the probability distribution is also for the current digit, however all the retangle encapsulated digts are classified as seen in the *Input Image*

The ROI with highest Y position (top) is selected as the *Current Digit* 

See demo:


https://github.com/user-attachments/assets/5b5d4e45-e76f-4ab9-97d3-6d46a1f03ccf

