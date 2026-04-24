## Description
In this competition, participants are required to build a system that can analyze PNG images containing deep learning multiple-choice questions (MCQs) and predict the correct answer.

Each image consists of:

A question related to deep learning concepts
Four answer options, with only one correct choice

Participants must design models capable of:

- Extracting text from images
- Understanding technical content
- Reasoning over the question and options
- Selecting the correct answer

⚠️ Note: No dataset will be provided. Participants are expected to create or source their own data, simulate similar question formats, or leverage external resources to train their models.

## Evaluation
Submissions will be evaluated using a scoring system with negative marking:

- +1 point for each correct answer
- −0.25 points for each incorrect answer
- 0 points for unanswered questions
- -1 point for hallucinated answer

Participants may use “5” to indicate a question is not answered.

The final score will be calculated as:

```
final_score = (number of correct option) - 0.25 x (number of incorrect option) -  (number of hallucinated value)
```

Where:

Predictions with values 1, 2, 3, 4 are treated as attempted answers
Prediction value 5 is treated as unanswered and receives no penalty
Any other output value will be treated as hallucinated value

This evaluation encourages both accuracy and strategic decision-making—participants may choose to skip uncertain questions.


## Submission File
A ```submission.csv``` must be submitted with the following format.

    id,image_name,option
    image_1,image_1,0
    image_2,image_2,0
    image_3,image_3,0
    etc.

The first columns ```id``` is same as ```image_name```.
For each question in the test set, you must predict the correct answer (or 5 for unanswered one). Any other value will be treated as hallucinated value.

## Dataset Description
The competition data is designed to simulate a real-world evaluation setup where the final test set remains hidden.

### Files Provided
* **test.csv** - The test set containing the list of test samples.
    - Column:
            ```image_id```: Same as image_name
            ```image_name```: Name of the image file corresponding to each question
* **image/** - folder containing PNG images of MCQs.
    - Each image corresponds to a row in test.csv
    - File names match the values in the image_name column
* **sample_submission.csv** - Sample submission file demonstrating the required format for submission.

### Hidden Test Set
* During the final evaluation and leaderboard scoring, the contents of the ```image/``` folder will be replaced with a hidden test dataset
* The structure will remain the same, but images will be different and not accessible beforehand
* This ensures a fair evaluation of model generalization
* At test time parent director to ```test.csv``` will be provided and thereafter the folder structure is same as the one provided in sample dataset
* Folder Structure
  ```
  .
  ├── images/
  │   ├── image_1.png
  │   └── image_2.png
  |   └── ...
  ├── test.csv
  └── sample_submission.csv

  ```

### Submission Requirement
Participants must submit a file named submission.csv:

* The format must strictly follow sample_submission.csv
* Required columns and structure are defined in the sample file and described on this page
* Each row should correspond to a question from test.csv

### Note:

* Do not rely on specific image content in the provided dataset, as it will change during evaluation
* Ensure your pipeline works generically for any valid input image
* All predictions must be generated based on the images referenced in test.csv

## Competition Rules:

- **Deadline to upload is 2nd May 23:59 UTC (3rd May 05:30AM IST)**
- **No Late Days are allowed for project submission**
- Notebook/Python file only competition (You'll upload link to the folder containing the jupyter notebook or python file on moodle and any model weights that is required)
- **Proper README to setup the environment without which you will be directly graded 0 (No communication from the TA will be done regarding this)**
- An environment.yml or requirements.txt file that can be used to create environment
- At most 50 questions will be provided 
- Runtime of the notebook shouldn't exceed 1 hr
- Your notebook will be run on 48GB L40s GPU
- Internet will be used only to setup the environment and no internet will be allowed on the final submitted notebook
- Do not cheat
- Okay to consult and discuss idea but final solution should be your own.
- Final grading will be based on final leaderboard standing
- Cite whatever source you'll be using in your notebook