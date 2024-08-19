# IBM-Project-2024

## Introduction

In today’s world, environmental consciousness is more important than ever. However, instilling these values in young children can be challenging. Enter [SortBot](https://github.com/Sibi-Agilan-17/SortBot), by harnessing the power of gamification, we can make the process of trash collection fun and engaging for students, encouraging them to participate actively in maintaining a clean environment. This project focuses on developing a trash collection system that not only promotes proper waste disposal but also rewards students for their participation, fostering lifelong habits of environmental stewardship.

## Project Aim

The aim of this project is to design and implement a gamified trash collection system that encourages school children to engage in proper waste disposal practices. By making the process interactive and rewarding, we aim to instill a sense of responsibility and environmental awareness in students, ultimately contributing to a cleaner school environment and promoting sustainable habits.

## Project description

The trash collection system will integrate gamification elements such as points, levels, and rewards to motivate students to dispose of waste correctly. Each student will be provided with an RFID card or a similar identification mechanism that they can use to log their waste disposal activities. The system will track the types and amounts of waste collected by each student, awarding points based on their actions.

Points can be earned for properly sorting waste into designated bins, with additional rewards for consistent participation or special challenges. The system will feature a leaderboard, allowing students to see their rankings compared to their peers, further encouraging competition and engagement. Rewards could include digital badges, certificates, or even tangible prizes that can be redeemed through the school’s reward program.

## Installation

 - Clone the github repo in your local device.
    ```bash
    git clone https://github.com/Sibi-Agilan-17/SortBot
   ```
 - Install the required libraries using the following command:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
    
### Preparing the data

 - The dataset to be used for training is in the folder `./dataset-original`. The dataset is in the form of images of trash items. The dataset is divided into 6 classes: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.
 - If you want to resize the images to a specific size, you can run the file `resize.py` to resize them. This will resize all the images in the `./dataset-original` folder to 256x256 and save them in the `./dataset-resized` folder.

### Training the model

 - To train the model, run the file `train.py`. This will train the model using the dataset in the `./dataset-resized` folder and save the model.
 - You can customise the training settings - by default the model is trained 10x on 10 epoch each.

### Testing the model

 - The model is tested simultaneously while training. The model is tested on the test dataset after each epoch and the accuracy is printed.

## Roadmap

### Phase 1: Research and Planning

Identify the target schools and conduct surveys to understand the current waste disposal habits of students.
Research existing gamification techniques and determine the most effective methods for this project.
Develop a detailed project plan, including timelines, required resources, and potential challenges.

### Phase 2: Design and Development

Design the hardware system, including RFID readers, waste bins with sensors, and a central server for data collection.
Develop the software interface, including the student dashboard, leaderboard, and reward system.
Integrate gamification elements into the software, ensuring a user-friendly experience for the students.

### Phase 3: Testing and Iteration

Conduct pilot tests in a select number of schools to gather feedback on the system’s usability and effectiveness.
Iterate on the design based on feedback, making improvements to both the hardware and software components.
Ensure the system is scalable and can be deployed in schools of varying sizes.

### Phase 4: Deployment and Training

Roll out the system across the selected schools, providing training to both students and teachers on how to use it.
Monitor the system’s performance and make adjustments as necessary.
Launch the rewards program and start tracking student participation.

### Phase 5: Evaluation and Expansion

Evaluate the system’s impact on students’ waste disposal habits and overall school cleanliness.
Collect feedback from students, teachers, and administrators to assess the project’s success.
Explore opportunities for expanding the system to more schools or incorporating additional environmental activities.

## Components

 - To be updated

## Conclusion

This gamified trash collection system represents an innovative approach to environmental education. By making waste disposal a fun and rewarding activity, we can effectively engage students in sustainable practices from a young age. Through careful planning, development, and execution, this project has the potential to significantly improve waste management in schools and inspire a new generation of environmentally conscious individuals.

## References
- [trashnet](https://github.com/garythung/trashnet) | dataset used for training the model
