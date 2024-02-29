#!/usr/bin/env python
# coding: utf-8


from vedo import *
import numpy as np


def RotationMatrix(theta, axis_name):
    """ calculate single rotation of $theta$ matrix around x,y or z
        code from: https://programming-surgeon.com/en/euler-angle-python-en/
    input
        theta = rotation angle(degrees)
        axis_name = 'x', 'y' or 'z'
    output
        3x3 rotation matrix
    """

    c = np.cos(theta * np.pi / 180)
    s = np.sin(theta * np.pi / 180)

    if axis_name == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, c, -s],
                                    [0, s, c]])
    if axis_name == 'y':
        rotation_matrix = np.array([[c, 0, s],
                                    [0, 1, 0],
                                    [-s, 0, c]])
    elif axis_name == 'z':
        rotation_matrix = np.array([[c, -s, 0],
                                    [s, c, 0],
                                    [0, 0, 1]])
    return rotation_matrix


def createCoordinateFrameMesh():
    """Returns the mesh representing a coordinate frame
    Args:
      No input args
    Returns:
      F: vedo.mesh object (arrows for axis)

    """
    _shaft_radius = 0.05
    _head_radius = 0.10
    _alpha = 1

    # x-axis as an arrow
    x_axisArrow = Arrow(start_pt=(0, 0, 0),
                        end_pt=(1, 0, 0),
                        s=None,
                        shaft_radius=_shaft_radius,
                        head_radius=_head_radius,
                        head_length=None,
                        res=12,
                        c='red',
                        alpha=_alpha)

    # y-axis as an arrow
    y_axisArrow = Arrow(start_pt=(0, 0, 0),
                        end_pt=(0, 1, 0),
                        s=None,
                        shaft_radius=_shaft_radius,
                        head_radius=_head_radius,
                        head_length=None,
                        res=12,
                        c='green',
                        alpha=_alpha)

    # z-axis as an arrow
    z_axisArrow = Arrow(start_pt=(0, 0, 0),
                        end_pt=(0, 0, 1),
                        s=None,
                        shaft_radius=_shaft_radius,
                        head_radius=_head_radius,
                        head_length=None,
                        res=12,
                        c='blue',
                        alpha=_alpha)

    originDot = Sphere(pos=[0, 0, 0],
                       c="black",
                       r=0.10)

    # Combine the axes together to form a frame as a single mesh object
    F = x_axisArrow + y_axisArrow + z_axisArrow + originDot

    return F


def getLocalFrameMatrix(R_ij, t_ij):
    """Returns the matrix representing the local frame
    Args:
      R_ij: rotation of Frame j w.r.t. Frame i
      t_ij: translation of Frame j w.r.t. Frame i
    Returns:
      T_ij: Matrix of Frame j w.r.t. Frame i.

    """
    # Rigid-body transformation [ R t ]
    T_ij = np.block([[R_ij, t_ij],
                     [np.zeros((1, 3)), 1]])

    return T_ij


def forward_kinematics(phi, L1, L2, L3, L4):
    R01 = RotationMatrix(phi[0], axis_name='z')
    p1 = np.array([[3], [2], [0.0]])
    t_01 = p1
    T_01 = getLocalFrameMatrix(R01, t_01)

    # Matrix of Frame 2 (written w.r.t. Frame 1, which is the previous frame)
    R_12 = RotationMatrix(phi[1], axis_name='z')  # Rotation matrix
    p2 = np.array([[L1 + 2 * 0.4], [0.0], [0.0]])  # Frame's origin (w.r.t. previous frame)
    t_12 = p2  # Translation vector

    # Matrix of Frame 2 w.r.t. Frame 1
    T_12 = getLocalFrameMatrix(R_12, t_12)

    # Matrix of Frame 2 w.r.t. Frame 0 (i.e., the world frame)
    T_02 = T_01 @ T_12

    R_23 = RotationMatrix(phi[2], axis_name='z')  # Rotation matrix
    p3 = np.array([[L2 + 2 * 0.4], [0.0], [0.0]])  # Frame's origin (w.r.t. previous frame)
    t_23 = p3  # Translation vector

    # Matrix of Frame 3 w.r.t. Frame 2
    T_23 = getLocalFrameMatrix(R_23, t_23)

    T_03 = T_01 @ T_12 @ T_23

    R_34 = RotationMatrix(phi[3], axis_name='z')
    p4 = np.array([[L3 + 0.4], [0.0], [0.0]])
    t_34 = p4
    T_34 = getLocalFrameMatrix(R_34, t_34)
    T_04 = T_01 @ T_12 @ T_23 @ T_34
    e = getLastPoint(T_04)
    return T_01, T_02, T_03, T_04, e


def displayAngle(length1, length2, angle1, angle2):
    # Set the limits of the graph x, y, and z ranges
    axes = Axes(xrange=(0, 20), yrange=(-2, 10), zrange=(0, 6))

    # Lengths of arm parts
    L1 = length1  # Length of link 1
    L2 = length2  # Length of link 2

    # Joint angles
    phi1 = angle1  # Rotation angle of part 1 in degrees
    phi2 = angle2  # Rotation angle of part 2 in degrees
    phi3 = 0  # Rotation angle of the end-effector in degrees
    T_01, T_02, T_03, T_04, e = forward_kinematics([phi1, phi2, phi3, 0], L1, L2, 0, 0)
    Frame1Arrows = createCoordinateFrameMesh()

    # Now, let's create a cylinder and add it to the local coordinate frame
    link1_mesh = Cylinder(r=0.4,
                          height=L1,
                          pos=(L1 / 2, 0, 0),
                          c="yellow",
                          alpha=.8,
                          axis=(1, 0, 0)
                          )

    # Also create a sphere to show as an example of a joint
    r1 = 0.4
    sphere1 = Sphere(r=r1).pos(-r1, 0, 0).color("gray").alpha(.8)

    # Combine all parts into a single object
    Frame1 = Frame1Arrows + link1_mesh + sphere1

    # Transform the part to position it at its correct location and orientation
    Frame1.apply_transform(T_01)

    # Create the coordinate frame mesh and transform
    Frame2Arrows = createCoordinateFrameMesh()

    # Now, let's create a cylinder and add it to the local coordinate frame
    link2_mesh = Cylinder(r=0.4,
                          height=L2,
                          pos=(L2 / 2, 0, 0),
                          c="red",
                          alpha=.8,
                          axis=(1, 0, 0)
                          )

    # Combine all parts into a single object
    Frame2 = Frame2Arrows + link2_mesh

    # Transform the part to position it at its correct location and orientation
    Frame2.apply_transform(T_02)

    # Create the coordinate frame mesh and transform. This point is the end-effector. So, I am
    # just creating the coordinate frame.
    Frame3 = createCoordinateFrameMesh()

    # Transform the part to position it at its correct location and orientation
    Frame3.apply_transform(T_03)
    return [Frame1, Frame2, Frame3, axes]


def getLastPoint(T0N):
    return T0N[:3, 3]


def drawforwardKinematics(phis, lengths, ranges):
    lengthone, lengthtwo = lengths
    phione, phitwo = phis
    stopone, stoptwo = ranges
    istepper = -1
    jstepper = 1
    j = phitwo
    i = phione
    while True:
        if i == stopone or i == phione + 1:
            istepper = istepper * -1
        if j == stoptwo or j == phitwo - 1:
            jstepper = jstepper * -1
        frames = displayAngle(lengthone, lengthtwo, i, j)
        frame = show([frames[0], frames[1], frames[2]], viewup="z", axes=frames[3], new=False, bg='beige', bg2='lb',
                     offscreen=False,
                     interactive=False)
        frame.remove(frames[0])
        frame.remove(frames[1])
        frame.remove(frames[2])
        i += istepper
        j += jstepper


if __name__ == '__main__':
    drawforwardKinematics([70, -50], [7, 2], [-20, 50])
