# contact_state_classification
This is the repo for the robotics project "contact state classification using machine learning".

#### Resources
Google Drive: https://drive.google.com/drive/folders/1GyiogHXgIxUiuVfkc2BFoljfxg-VJTo3
Datasets e.g., RoboticsProject2510.pkl and demo video e.g., wiggling_example.mp4 are saved here.
#### Feature Value Explanation 
| Feature     | Description |
| :---        |    :----:   |
| label      | label for contact state       |
| lock_type   | different type of latch lock        |
| init_q   | initial joint state        |
| init_base_ht_ee   | initial end-effector(ee) pose in the Cartesian space on the robot base frame        |
| init_image   | None |
| init_rs_image   | initial image on the robot end-effector from a realsense(rs) camera      |
| init_usb_image   | initial image from an external perspective from a usb camera       |
| desired control values             |
| d_ee_velocity   | desired ee velocity on the ee frame      |
| d_ee_theta   | desired ee velocity but in the spherical coordinate        |
| d_ee_phi   | desired ee velocity but in the spherical coordinate        |
| observed state values             |
| dist   | observed moved distance after execution an exploration action (currently 12 actions for one exploration behavior in total)       |
| base_s_ee   | observed moved distance as form of directed vector after execution an exploration action        |
| error_q   | joint value difference. Once the robot finished a exploration action, it should move back to the initial joint pose (init_q). This feature shows how good the robot moved back to the initial pose       |
| obs_ee_theta   | observed ee velocity in the spherical coordinate after sending the d_ee_velocity to the robot        |
| obs_ee_phi   | observed ee velocity in the spherical coordinate after sending the d_ee_velocity to the robot        |
| init_ee_ft_robot   | observed force/torque sensor reading on the ee frame <strong> before </strong> controlling one exploration action         |
| curt_ee_ft_robot   | observed force/torque sensor reading on the ee frame <strong> after </strong> controlling one exploration action         |

### Background Knowledge
* Panda Robot Frame Information: 
  https://frankaemika.github.io/docs/control_parameters.html
  
  Frame 0 = Base Frame
  
  Frame F = End-Effector Frame
  
* Spherical Coordinate:
https://upload.wikimedia.org/wikipedia/commons/d/dc/3D_Spherical_2.svg
  
  We transform the translation (x, y, z) into spherical coordinate to reduce the dimensionality from 3(x, y, z) to 2 (theta, phi). As we are only interested in moving direction namely theta and phi

### TODO List

- [x] Simple Dataset (Only for one lock and one grasp pose, named RoboticsProject2510.pkl in the Google cloud)
- [x] Simple Classifier  
- [x] Doc/Explanation for feature values
- [ ] Rapidly Visualization Tool
  - [ ] Easy modify and visualize the dataset
  - [ ] Candidate tools: Facets, Visdom
- [x] Dataset includes multi locks and grasp poses
  - [ ] two different latch locks
  - [ ] three different grasp poses

### Project Plan
- [ ] 02.11-08.11 Feature preprocessing + Literature Review/ Research(parallel work)
- [ ] 09.11-15.11 Feature preprocessing + Feature engineering
- [ ] 16.11-22.11 Feature engineering + visualization of the original data
- [ ] 23.11-29.11 Implementation of KNN classifier + verification + visualization
- [ ] 30.11-06.12 Reasoning for KNN + documentation + Testing
- [ ] 07.12-13.12 Research of State of other art classifiers, find 3 possible best candidates
- [ ] Break
- [ ] 04.01-10.01 Implementing 1st best classifier + validation
- [ ] 11.01-17.01 Visualization + reasoning
- [ ] 18.01-24.01 Implementing 2nd best classifier
- [ ] 25.01-31.01 Implementing last best classifier (optional)
- [ ] 01.02-07.02 Final Presentation + Testing on real robot, hopefully :) 
- [ ] 08.02-14.02 Results, documentation and The report / Presentation
