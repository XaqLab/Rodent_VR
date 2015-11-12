
"""
Run using SLSQP and starting with the parameters from 2015-02-10
Parameters
Projector y: 0.795981693151
Projector z: 0.124542772038
Projector theta: 0.399999989467
Projector vertical offset: 0.0882189101299
Projector roll: 0.0
Mirror radius: 0.292383065916
Dome y-coordinate: 0.109692750494
Dome z-coordinate: 0.499999999896
Dome radius: 0.500000000082
Animal y-coordinate: -0.299999999914
Animal z-coordinate: 0.300000045833

[actual pitch, actual yaw], [estimated pitch, estimated yaw], estimated - actual
[30.522, -9.750], [27.650,  0.000], [-2.872,  9.750]
[38.766, 34.991], [29.862, 22.870], [-8.903, -12.120]
[40.645, 68.944], [37.824, 34.012], [-2.821, -34.932]
[15.298, -5.703], [15.325,  0.000], [ 0.027,  5.703]
[15.216, 31.426], [15.819, 24.111], [ 0.604, -7.315]
[15.231, 22.193], [25.718, 44.220], [10.487, 22.026]
[ 0.000,  0.000], [-0.007,  0.000], [-0.007,  0.000]
[-0.275, 18.604], [-0.693, 23.655], [-0.418,  5.051]
[-0.519, 34.647], [ 1.634, 43.310], [ 2.153,  8.663]

Sum of errors: 1.7208125225
"""
animal_position = [0, -0.3, 0.3]
dome_center = [0, 0.110, 0.5]
dome_radius = 0.5
mirror_radius = 0.292
projector_focal_point = [0.0, 0.796, 0.125]
projector_roll = 0.0
projector_theta = 0.4
projector_vertical_offset = 0.088
parameters = dict(animal_position = animal_position,
                  dome_center = dome_center,
                  dome_radius = dome_radius,
                  projector_roll = projector_roll,
                  mirror_radius = mirror_radius,
                  projector_focal_point = projector_focal_point,
                  projector_theta = projector_theta,
                  projector_vertical_offset = projector_vertical_offset)
