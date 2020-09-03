import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_keypoints(color_image, keypoints_2d, frame_number, text):
    for kp in keypoints_2d:
        cv2.circle(color_image, (int(kp[0]), int(kp[1])), radius=0, color=(0, 0, 255), thickness=10)

    cv2.putText(color_image, f'Frame: {frame_number} - {text}', (10,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

    return color_image


def draw_stick_figure_op(keypoints, azimuth, elavation):

    x_points, y_points, z_points = [],[], []

    for k in keypoints:
        x_points.append(k[0])
        y_points.append(k[1])
        z_points.append(k[2])

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.view_init(elavation, azimuth)
   

    ax.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5,0.5))

    #plt.axis('off')

    ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv')
    ax.scatter3D(x_points, y_points, z_points, cmap='hsv')
    ax.plot([x_points[0], x_points[1]], [y_points[0], y_points[1]], [z_points[0], z_points[1]])
    ax.plot([x_points[1], x_points[2]], [y_points[1], y_points[2]], [z_points[1], z_points[2]])
    ax.plot([x_points[1], x_points[5]], [y_points[1], y_points[5]], [z_points[1], z_points[5]])
    ax.plot([x_points[1], x_points[8]], [y_points[1], y_points[8]], [z_points[1], z_points[8]])
    ax.plot([x_points[8], x_points[9]], [y_points[8], y_points[9]], [z_points[8], z_points[9]])
    ax.plot([x_points[9], x_points[10]], [y_points[9], y_points[10]], [z_points[9], z_points[10]])
    ax.plot([x_points[10], x_points[11]], [y_points[10], y_points[11]], [z_points[10], z_points[11]])
    ax.plot([x_points[12], x_points[13]], [y_points[12], y_points[13]], [z_points[12], z_points[13]])
    ax.plot([x_points[13], x_points[14]], [y_points[13], y_points[14]], [z_points[13], z_points[14]])
    ax.plot([x_points[2], x_points[3]], [y_points[2], y_points[3]], [z_points[2], z_points[3]])
    ax.plot([x_points[3], x_points[4]], [y_points[3], y_points[4]], [z_points[3], z_points[4]])
    ax.plot([x_points[5], x_points[6]], [y_points[5], y_points[6]], [z_points[5], z_points[6]])
    ax.plot([x_points[6], x_points[7]], [y_points[6], y_points[7]], [z_points[6], z_points[7]])
    ax.plot([x_points[8], x_points[12]], [y_points[8], y_points[12]], [z_points[8], z_points[12]])
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    cv2.putText(img, 'OpenPose 3D', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)

    plt.close()

    return img


def draw_stick_figure_vp(keypoints, azimuth, elavation):

    x_points, y_points, z_points = [],[], []

    for k in keypoints:
        x_points.append(k[0])
        y_points.append(k[1])
        z_points.append(k[2])

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.view_init(elavation, azimuth)


    ax.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5,0.5))

    plt.axis('off')

    ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv')
    ax.scatter3D(x_points, y_points, z_points, cmap='hsv')
    ax.plot([x_points[0], x_points[1]], [y_points[0], y_points[1]], [z_points[0], z_points[1]])
    ax.plot([x_points[1], x_points[2]], [y_points[1], y_points[2]], [z_points[1], z_points[2]])
    ax.plot([x_points[2], x_points[3]], [y_points[2], y_points[3]], [z_points[2], z_points[3]])
    ax.plot([x_points[0], x_points[4]], [y_points[0], y_points[4]], [z_points[0], z_points[4]])
    ax.plot([x_points[4], x_points[5]], [y_points[4], y_points[5]], [z_points[4], z_points[5]])
    ax.plot([x_points[5], x_points[6]], [y_points[5], y_points[6]], [z_points[5], z_points[6]])
    ax.plot([x_points[0], x_points[7]], [y_points[0], y_points[7]], [z_points[0], z_points[7]])
    ax.plot([x_points[7], x_points[8]], [y_points[7], y_points[8]], [z_points[7], z_points[8]])
    ax.plot([x_points[8], x_points[9]], [y_points[8], y_points[9]], [z_points[8], z_points[9]])
    ax.plot([x_points[8], x_points[11]], [y_points[8], y_points[11]], [z_points[8], z_points[11]])
    ax.plot([x_points[9], x_points[10]], [y_points[9], y_points[10]], [z_points[9], z_points[10]])
    ax.plot([x_points[11], x_points[12]], [y_points[11], y_points[12]], [z_points[11], z_points[12]])
    ax.plot([x_points[12], x_points[13]], [y_points[12], y_points[13]], [z_points[12], z_points[13]])
    ax.plot([x_points[8], x_points[14]], [y_points[8], y_points[14]], [z_points[8], z_points[14]])
    ax.plot([x_points[14], x_points[15]], [y_points[14], y_points[15]], [z_points[14], z_points[15]])
    ax.plot([x_points[15], x_points[16]], [y_points[15], y_points[16]], [z_points[15], z_points[16]])

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    cv2.putText(img, 'VideoPose', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)


    plt.close()

    return img
