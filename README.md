# AA228_FinalProject_Fall2023

This repository includes the files for our AA228 final project. For this project, we provide an example and an evaluation of an intelligent tutoring system modeled by a POMDP that uses an offline QMDP approach and a discrete state filter belief update in order to determine the next approximately optimal interaction with the student.

A state in our model is defined as the probability of student understanding. We discretize our state space to simplify our approach. 

To run our algorithm and evaluation in terminal: 
python3 main.py number_of_discrete_states initial_simulated_student_understanding 

where number_of_discrete_states refers to how you want to discretize the student understanding state space and initial_simulated_student_understanding refers to the initial probability that a simulated student will get an answer correct or not. 

FinalProject.py includes our code for initializing our POMDP model and creating alpha vectors via a QMDP approach. 

Simulate.py includes our code to simulate student responses, update student understanding, and update our tutor's belief of student understanding. 

main.py runs the tutoring system, calling functions in FinalProject.py and Simulate.py and keeping track of the number of questions and hints the tutoring system provides to the simulated student. 
