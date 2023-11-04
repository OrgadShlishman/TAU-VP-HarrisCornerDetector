# TAU-VP-HarrisCornerDetector
Harris Corner Detector

# Background
From Wikipedia, the free encyclopedia
The Harris corner detector is a corner detection operator that is commonly used in computer vision algorithms to extract corners and infer features of an image. It was first introduced by Chris Harris and Mike Stephens in 1988 upon the improvement of Moravec's corner detector.[1] Compared to its predecessor, Harris' corner detector takes the differential of the corner score into account with reference to direction directly, instead of using shifting patches for every 45 degree angles, and has been proved to be more accurate in distinguishing between edges and corners.[2] Since then, it has been improved and adopted in many algorithms to preprocess images for subsequent applications.

Introduction
A corner is a point whose local neighborhood stands in two dominant and different edge directions. In other words, a corner can be interpreted as the junction of two edges, where an edge is a sudden change in image brightness.[3] Corners are the important features in the image, and they are generally termed as interest points which are invariant to translation, rotation and illumination. Although corners are only a small percentage of the image, they contain the most important features in restoring image information, and they can be used to minimize the amount of processed data for motion tracking, image stitching, building 2D mosaics, stereo vision, image representation and other related computer vision areas.

In order to capture the corners from the image, researchers have proposed many different corner detectors including the Kanade-Lucas-Tomasi (KLT) operator and the Harris operator which are most simple, efficient and reliable for use in corner detection. These two popular methodologies are both closely associated with and based on the local structure matrix. Compared to the Kanade-Lucas-Tomasi corner detector, the Harris corner detector provides good repeatability under changing illumination and rotation, and therefore, it is more often used in stereo matching and image database retrieval. Although there still exists drawbacks and limitations, the Harris corner detector is still an important and fundamental technique for many computer vision applications.

Development of Harris corner detection algorithm [1]
Without loss of generality, we will assume a grayscale 2-dimensional image is used. Let this image be given by 
�
I. Consider taking an image patch 
(
�
,
�
)
∈
�
{\displaystyle (x,y)\in W}(window) and shifting it by 
(
Δ
�
,
Δ
�
)
{\displaystyle (\Delta x,\Delta y)}. The sum of squared differences (SSD) between these two patches, denoted 
�
f, is given by:

�
(
Δ
�
,
Δ
�
)
=
∑
(
�
�
,
�
�
)
∈
�
(
�
(
�
�
,
�
�
)
−
�
(
�
�
+
Δ
�
,
�
�
+
Δ
�
)
)
2
{\displaystyle f(\Delta x,\Delta y)={\underset {(x_{k},y_{k})\in W}{\sum }}\left(I(x_{k},y_{k})-I(x_{k}+\Delta x,y_{k}+\Delta y)\right)^{2}}
�
(
�
+
Δ
�
,
�
+
Δ
�
)
{\displaystyle I(x+\Delta x,y+\Delta y)} can be approximated by a Taylor expansion. Let 
�
�
I_{x} and 
�
�
 I_y be the partial derivatives of 
�
 I, such that

�
(
�
+
Δ
�
,
�
+
Δ
�
)
≈
�
(
�
,
�
)
+
�
�
(
�
,
�
)
Δ
�
+
�
�
(
�
,
�
)
Δ
�
{\displaystyle I(x+\Delta x,y+\Delta y)\approx I(x,y)+I_{x}(x,y)\Delta x+I_{y}(x,y)\Delta y}
This produces the approximation

�
(
Δ
�
,
Δ
�
)
≈
∑
(
�
,
�
)
∈
�
(
�
�
(
�
,
�
)
Δ
�
+
�
�
(
�
,
�
)
Δ
�
)
2
,
{\displaystyle f(\Delta x,\Delta y)\approx {\underset {(x,y)\in W}{\sum }}\left(I_{x}(x,y)\Delta x+I_{y}(x,y)\Delta y\right)^{2},}
which can be written in matrix form:

�
(
Δ
�
,
Δ
�
)
≈
(
Δ
�
Δ
�
)
�
(
Δ
�
Δ
�
)
,
{\displaystyle f(\Delta x,\Delta y)\approx {\begin{pmatrix}\Delta x&\Delta y\end{pmatrix}}M{\begin{pmatrix}\Delta x\\\Delta y\end{pmatrix}},}
where M is the structure tensor,

�
=
∑
(
�
,
�
)
∈
�
[
�
�
2
�
�
�
�
�
�
�
�
�
�
2
]
=
[
∑
(
�
,
�
)
∈
�
�
�
2
∑
(
�
,
�
)
∈
�
�
�
�
�
∑
(
�
,
�
)
∈
�
�
�
�
�
∑
(
�
,
�
)
∈
�
�
�
2
]
{\displaystyle M={\underset {(x,y)\in W}{\sum }}{\begin{bmatrix}I_{x}^{2}&I_{x}I_{y}\\I_{x}I_{y}&I_{y}^{2}\end{bmatrix}}={\begin{bmatrix}{\underset {(x,y)\in W}{\sum }}I_{x}^{2}&{\underset {(x,y)\in W}{\sum }}I_{x}I_{y}\\{\underset {(x,y)\in W}{\sum }}I_{x}I_{y}&{\underset {(x,y)\in W}{\sum }}I_{y}^{2}\end{bmatrix}}}
Process of Harris corner detection algorithm[4][5][6]
Commonly, Harris corner detector algorithm can be divided into five steps.

Color to grayscale
Spatial derivative calculation
Structure tensor setup
Harris response calculation
Non-maximum suppression
Color to grayscale
If we use Harris corner detector in a color image, the first step is to convert it into a grayscale image, which will enhance the processing speed.

The value of the gray scale pixel can be computed as a weighted sums of the values R, B and G of the color image,

∑
�
∈
{
�
,
�
,
�
}
�
�
⋅
�
{\displaystyle \sum _{C\,\in \,\{R,G,B\}}w_{C}\cdot C},
where, e.g.,

�
�
=
0.299
,
 
�
�
=
0.587
,
 
�
�
=
1
−
(
�
�
+
�
�
)
=
0.114.
{\displaystyle w_{R}=0.299,\ w_{G}=0.587,\ w_{B}=1-(w_{R}+w_{G})=0.114.}
Spatial derivative calculation
Next, we are going to find the derivative with respect to x and the derivative with respect to y, 
�
�
(
�
,
�
)
{\displaystyle I_{x}(x,y)} and 
�
�
(
�
,
�
)
{\displaystyle I_{y}(x,y)}.

Structure tensor setup
With 
�
�
(
�
,
�
)
{\displaystyle I_{x}(x,y)} , 
�
�
(
�
,
�
)
{\displaystyle I_{y}(x,y)}, we can construct the structure tensor 
�
M.

Harris response calculation
For 
�
≪
�
{\displaystyle x\ll y}, one has 
�
⋅
�
�
+
�
=
�
1
1
+
�
/
�
≈
�
.
{\displaystyle {\tfrac {x\cdot y}{x+y}}=x{\tfrac {1}{1+x/y}}\approx x.} In this step, we compute the smallest eigenvalue of the structure tensor using that approximation:

�
min
≈
�
1
�
2
(
�
1
+
�
2
)
=
det
(
�
)
tr
⁡
(
�
)
{\displaystyle \lambda _{\min }\approx {\frac {\lambda _{1}\lambda _{2}}{(\lambda _{1}+\lambda _{2})}}={\frac {\det(M)}{\operatorname {tr} (M)}}}
with the trace 
t
r
(
�
)
=
�
11
+
�
22
{\displaystyle \mathrm {tr} (M)=m_{11}+m_{22}}.

Another commonly used Harris response calculation is shown as below,

�
=
�
1
�
2
−
�
(
�
1
+
�
2
)
2
=
det
(
�
)
−
�
tr
⁡
(
�
)
2
{\displaystyle R=\lambda _{1}\lambda _{2}-k(\lambda _{1}+\lambda _{2})^{2}=\det(M)-k\operatorname {tr} (M)^{2}}

where 
�
k is an empirically determined constant; 
�
∈
[
0.04
,
0.06
]
{\displaystyle k\in [0.04,0.06]} .
